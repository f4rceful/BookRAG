from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator, Callable
import asyncio
import math
import os
import re
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Примерное количество символов на одну страницу книги
_CHARS_PER_PAGE = 1800

# Шаблон для поиска структурных заголовков книги (Том, Часть, Глава)
_STRUCTURE_RE = re.compile(
    r'(?P<tome>^(?:ТОМ|Том)\s+\S+)'
    r'|(?P<part>^(?:ЧАСТЬ|Часть)\s+\S+)'
    r'|(?P<chapter>^(?:ГЛАВА|Глава)\s+\S+)'
    r'|(?P<epilogue>^(?:ЭПИЛОГ|Эпилог)\s*)',
    re.MULTILINE | re.UNICODE
)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_STOPWORDS = {
    "а", "без", "был", "была", "были", "быть", "в", "во", "вот", "все", "вы", "где", "да", "для",
    "до", "его", "ее", "её", "если", "есть", "же", "за", "и", "из", "или", "им", "их", "к", "как",
    "какой", "какая", "какие", "когда", "кто", "ли", "мне", "мы", "на", "над", "не", "него", "нее",
    "неё", "но", "ну", "о", "об", "он", "она", "они", "оно", "от", "по", "под", "при", "с", "со",
    "так", "там", "те", "то", "того", "тоже", "той", "только", "том", "ты", "у", "уж", "что", "чтобы",
    "эта", "это", "этот", "эту", "я",
}
_ENDING_QUERY_TERMS = {
    "эпилог", "эпилоге", "финал", "финале", "конец", "конце", "заканчивается",
    "заканчивает", "заканчивают", "последний", "последняя", "последние", "последних",
}


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 7:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU поддерживается, используем CUDA: {device_name}")
                return "cuda"
            else:
                device_name = torch.cuda.get_device_name(0)
                logger.warning(f"GPU {device_name} (sm_{cap[0]}{cap[1]}) не поддерживается PyTorch, используем CPU")
    except Exception:
        pass
    logger.info("Используем CPU для эмбеддингов")
    return "cpu"


class RAGService:
    def __init__(self):
        logger.info(f"Инициализация эмбеддингов: {settings.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": _get_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._bm25_preprocess_func = self._build_bm25_preprocess_func()
        self._indexing_progress: Dict[str, Dict] = {}

        # Абсолютный путь для работы с ChromaDB на Windows
        persist_dir = os.path.abspath(settings.chroma_persist_directory)
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"Инициализация ChromaDB: {persist_dir}")
        self.vector_store = Chroma(
            collection_name="books_collection",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Инициализация Ollama: {settings.model_name} на {settings.ollama_base_url}")
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.model_name,
            temperature=0.1
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True
        )

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты — умный литературный ассистент. Используй предоставленные фрагменты книги как основу для своего ответа.

Тебе разрешено:
- Анализировать и сопоставлять информацию из разных фрагментов
- Делать логические выводы, которые прямо следуют из содержания текста
- Описывать мотивы персонажей и причинно-следственные связи на основе фрагментов
- Синтезировать целостный ответ, объединяя несколько фрагментов

Тебе запрещено:
- Выдумывать факты, которых нет ни в одном из фрагментов
- Добавлять внешние знания, не относящиеся к тексту книги
- Смешивать события из разных временных периодов книги

Если у фрагментов есть метки Том/Часть/Глава/Эпилог — используй их для понимания хронологии событий.
Если фрагменты не содержат достаточно информации для ответа, скажи: «К сожалению, в загруженных материалах недостаточно информации для ответа на этот вопрос.»

Фрагменты текста (Контекст):
{context}"""),
            ("user", "{question}")
        ])

        self._query_expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помогаешь улучшить поиск по книгам.
Переформулируй вопрос пользователя 2 разными способами — короткими конкретными фразами,
которые могут встретиться в художественном тексте.
Верни ТОЛЬКО 2 строки без нумерации и объяснений."""),
            ("user", "{question}")
        ])

        self._bm25_doc_count: int = -1
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._source_chunks_cache: Dict[str, List[Document]] = {}
        self._source_chunk_positions: Dict[str, Dict[int, int]] = {}
        self._bm25_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._current_model_name = settings.model_name
        # Ленивая загрузка ре-ранкера
        self._reranker_tried: bool = False
        self._reranker = None
        self._index_lock = threading.Lock()

    def _load_reranker(self):
        # Загрузка Cross-Encoder для ре-ранкинга при первом вызове
        if self._reranker_tried:
            return self._reranker
        self._reranker_tried = True
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Загрузка Cross-Encoder для ре-ранкинга...")
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2')
            logger.info("Cross-Encoder готов")
        except Exception as e:
            logger.warning(f"Cross-Encoder недоступен, используется базовое ранжирование: {e}")
        return self._reranker

    def _expand_query(self, question: str) -> List[str]:
        # Расширение запроса через LLM
        try:
            with self._model_lock:
                llm = self.llm
            resp = llm.invoke(
                self._query_expansion_prompt.format_messages(question=question)
            )
            variants = [line.strip() for line in resp.content.strip().split('\n') if line.strip()]
            # Удаление возможной нумерации из ответа LLM
            clean_variants = []
            for v in variants:
                clean_v = re.sub(r'^\d+[\.\)\s]+', '', v).strip()
                if clean_v:
                    clean_variants.append(clean_v)
            return [question] + clean_variants[:2]
        except Exception as e:
            logger.warning(f"Query expansion недоступен: {e}")
            return [question]

    @staticmethod
    def _reranker_score_to_display(raw_score: float) -> float:
        # Преобразование оценки Cross-Encoder в диапазон [0, 1]
        return round(1.0 / (1.0 + math.exp(-raw_score / 3.0)), 3)

    @staticmethod
    def _basic_bm25_tokens(text: str) -> List[str]:
        return [
            token.lower()
            for token in _TOKEN_RE.findall(text)
            if len(token) > 2 and token.lower() not in _STOPWORDS
        ]

    @staticmethod
    def _fallback_russian_stem(token: str) -> str:
        # Упрощенный стемминг для русского языка
        suffixes = (
            "иями", "ями", "ами", "его", "ого", "ему", "ому", "ыми", "ими",
            "ией", "ей", "ий", "ый", "ой", "ая", "яя", "ое", "ее",
            "ам", "ям", "ах", "ях", "ом", "ем", "ов", "ев", "ёв",
            "ую", "юю", "ия", "ья", "ие", "ье", "ий", "ый", "ой",
            "а", "я", "ы", "и", "е", "у", "ю", "о",
        )
        normalized = token.lower().replace("ё", "е")
        if len(normalized) <= 4 or not re.search(r"[а-я]", normalized):
            return normalized
        for suffix in suffixes:
            if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 3:
                return normalized[: -len(suffix)]
        return normalized

    def _build_bm25_preprocess_func(self) -> Callable[[str], List[str]]:
        # Настройка препроцессора для BM25 с поддержкой русского языка
        try:
            from nltk.stem.snowball import SnowballStemmer

            stemmer = SnowballStemmer("russian")
            logger.info("BM25 использует SnowballStemmer для русского языка")

            def preprocess(text: str) -> List[str]:
                return [stemmer.stem(token) for token in self._basic_bm25_tokens(text)]

            return preprocess
        except Exception as exc:
            logger.warning(
                "NLTK недоступен, BM25 использует встроенный русский стеммер: %s",
                exc,
            )

            def preprocess(text: str) -> List[str]:
                return [self._fallback_russian_stem(token) for token in self._basic_bm25_tokens(text)]

            return preprocess

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, int]:
        return doc.metadata.get("source", ""), int(doc.metadata.get("start_index", 0))

    def _query_tokens(self, query: str) -> List[str]:
        return self._bm25_preprocess_func(query)

    def _lexical_overlap_score(self, query_tokens: List[str], doc: Document) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = set(self._bm25_preprocess_func(doc.page_content))
        if not doc_tokens:
            return 0.0
        overlap = sum(1 for token in set(query_tokens) if token in doc_tokens)
        return overlap / max(len(set(query_tokens)), 1)

    def _has_ending_intent(self, query_tokens: List[str], raw_query: str) -> bool:
        raw_query_lower = raw_query.lower()
        if any(term in raw_query_lower for term in _ENDING_QUERY_TERMS):
            return True
        return any(token in _ENDING_QUERY_TERMS for token in query_tokens)

    @staticmethod
    def _ending_position_boost(doc: Document, max_start_index: int) -> float:
        if max_start_index <= 0:
            return 0.0
        start_index = int(doc.metadata.get("start_index", 0))
        relative_pos = start_index / max_start_index
        content = doc.page_content.lower()
        boost = max(0.0, (relative_pos - 0.72) * 2.4)
        if "к о н е ц" in content or "конец" in content:
            boost += 0.6
        return boost

    @staticmethod
    def _detect_source_version(text: str) -> str:
        head = text[:2000].lower()
        if "первый вариант романа" in head:
            return "first_draft"
        return "standard"

    def _rebuild_source_cache(self, docs: List[Document]) -> None:
        grouped: Dict[str, List[Document]] = {}
        for doc in docs:
            source = doc.metadata.get("source", "")
            grouped.setdefault(source, []).append(doc)

        self._source_chunks_cache = {}
        self._source_chunk_positions = {}
        for source, source_docs in grouped.items():
            sorted_docs = sorted(source_docs, key=lambda d: int(d.metadata.get("start_index", 0)))
            self._source_chunks_cache[source] = sorted_docs
            self._source_chunk_positions[source] = {
                int(doc.metadata.get("start_index", 0)): index
                for index, doc in enumerate(sorted_docs)
            }

    def _get_source_neighbors(self, doc: Document, radius: int = 1) -> List[Document]:
        source = doc.metadata.get("source", "")
        start_index = int(doc.metadata.get("start_index", 0))
        source_docs = self._source_chunks_cache.get(source)
        source_positions = self._source_chunk_positions.get(source)
        if not source_docs or not source_positions or start_index not in source_positions:
            return []

        center = source_positions[start_index]
        neighbors: List[Document] = []
        for idx in range(max(0, center - radius), min(len(source_docs), center + radius + 1)):
            if idx == center:
                continue
            neighbors.append(source_docs[idx])
        return neighbors

    def _get_source_tail_docs(self, tail_size: int = 8) -> List[Document]:
        tail_docs: List[Document] = []
        for docs in self._source_chunks_cache.values():
            tail_docs.extend(docs[-tail_size:])
        return tail_docs

    def _build_context_notes(self, question: str, search_results: List[Dict[str, Any]]) -> List[str]:
        question_lower = question.lower()
        notes: List[str] = []
        if "эпилог" in question_lower:
            for result in search_results:
                source_version = result.get("source_version", "")
                if source_version == "first_draft":
                    source = result.get("source", "источник")
                    note = (
                        f"Загруженная версия '{source}' помечена как 'Первый вариант романа' "
                        "и не содержит отдельного раздела 'Эпилог'."
                    )
                    if note not in notes:
                        notes.append(note)
        return notes

    @staticmethod
    def _parse_structure_markers(text: str) -> list:
        # Поиск заголовков (Том, Часть, Глава) в тексте
        markers = []
        for m in _STRUCTURE_RE.finditer(text):
            if m.group('tome'):
                markers.append((m.start(), 'tome', m.group('tome').strip()))
            elif m.group('part'):
                markers.append((m.start(), 'part', m.group('part').strip()))
            elif m.group('chapter'):
                markers.append((m.start(), 'chapter', m.group('chapter').strip()))
            elif m.group('epilogue'):
                markers.append((m.start(), 'epilogue', m.group('epilogue').strip()))
        return markers

    @staticmethod
    def _get_chunk_structure(markers: list, chunk_start: int) -> dict:
        # Определение структуры книги для текущего фрагмента
        tome = part = chapter = epilogue = ""
        for pos, mtype, value in markers:
            if pos > chunk_start:
                break
            if mtype == 'tome':
                tome, part, chapter, epilogue = value, "", "", ""
            elif mtype == 'part':
                part, chapter, epilogue = value, "", ""
            elif mtype == 'chapter':
                chapter, epilogue = value, ""
            else:
                chapter = ""
                epilogue = value
        return {
            k: v
            for k, v in [('tome', tome), ('part', part), ('chapter', chapter), ('epilogue', epilogue)]
            if v
        }

    def _get_bm25_retriever(self, k: int) -> Optional[BM25Retriever]:
        # Получение BM25-ретривера с обновлением при изменении базы
        with self._bm25_lock:
            try:
                current_count = self.vector_store._collection.count()
            except Exception:
                return None

            if current_count == 0:
                return None

            if current_count != self._bm25_doc_count:
                logger.info(f"Пересборка BM25-индекса ({current_count} чанков)...")
                data = self.vector_store.get(include=["documents", "metadatas"])
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(data["documents"], data["metadatas"])
                ]
                self._rebuild_source_cache(docs)
                self._bm25_retriever = BM25Retriever.from_documents(
                    docs,
                    preprocess_func=self._bm25_preprocess_func,
                )
                self._bm25_doc_count = current_count
                logger.info("BM25-индекс готов")

            self._bm25_retriever.k = k
            return self._bm25_retriever

    def set_model(self, model_name: str) -> Dict[str, str]:
        # Смена текущей модели Ollama
        with self._model_lock:
            logger.info(f"Смена модели на: {model_name}")
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=model_name,
                temperature=0.1
            )
            self._current_model_name = model_name
        return {"status": "success", "model": model_name}

    def get_current_model(self) -> str:
        return self._current_model_name

    def get_books(self) -> List[str]:
        # Получение списка всех книг в базе
        results = self.vector_store.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sorted(list(sources))

    def get_indexing_progress(self) -> Dict[str, Dict]:
        # Прогресс индексации активных задач
        return dict(self._indexing_progress)

    async def index_document_async(self, text: str, filename: str) -> AsyncGenerator[Dict[str, Any], None]:
        # Асинхронная индексация документа с трансляцией прогресса
        indexed_any = False
        success = False
        
        # Предварительная очистка данных при повторной попытке
        try:
            existing = await asyncio.to_thread(self.vector_store.get, where={"source": filename}, include=["metadatas"])
            if existing.get("ids") and len(existing["ids"]) > 0:
                logger.info(f"Обнаружены частичные данные для '{filename}'. Очистка перед новой попыткой...")
                await asyncio.to_thread(self.delete_document, filename)
        except Exception as e:
            logger.warning(f"Ошибка при предварительной очистке '{filename}': {e}")

        try:
            with self._index_lock:
                logger.info(f"Начало индексации документа: {filename}")

                structure_markers = self._parse_structure_markers(text)
                doc = Document(page_content=text, metadata={"source": filename})
                chunks = await asyncio.to_thread(self.text_splitter.split_documents, [doc])
                
                # Исключение слишком коротких фрагментов
                MIN_INDEX_CHUNK_LENGTH = 100
                chunks = [
                    c for c in chunks
                    if len(c.page_content.strip()) >= MIN_INDEX_CHUNK_LENGTH
                ]
                logger.info(f"[{filename}] После фильтрации коротких чанков: {len(chunks)}")
                
                source_version = self._detect_source_version(text)

                total_chunks = len(chunks)
                self._indexing_progress[filename] = {"percent": 0, "current": 0, "total": total_chunks}
                yield {
                    "type": "start",
                    "filename": filename,
                    "total_chunks": total_chunks
                }

                for index, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_index": index,
                        "source_total_chunks": total_chunks,
                        "source_total_chars": len(text),
                        "source_version": source_version,
                    })
                    if structure_markers:
                        structure = self._get_chunk_structure(
                            structure_markers, chunk.metadata.get("start_index", 0)
                        )
                        chunk.metadata.update(structure)

                for i in range(0, total_chunks, settings.index_batch_size):
                    batch = chunks[i:i + settings.index_batch_size]
                    await asyncio.to_thread(self.vector_store.add_documents, batch)
                    indexed_any = True
                    
                    current_count = min(i + settings.index_batch_size, total_chunks)
                    pct = round((current_count / total_chunks) * 100)
                    self._indexing_progress[filename] = {"percent": pct, "current": current_count, "total": total_chunks}
                    yield {
                        "type": "progress",
                        "current": current_count,
                        "total": total_chunks,
                        "percent": pct
                    }
                    await asyncio.sleep(0.05)

                self._bm25_doc_count = -1
                self._source_chunks_cache = {}
                self._source_chunk_positions = {}

                success = True
                self._indexing_progress.pop(filename, None)
                yield {
                    "type": "success",
                    "filename": filename,
                    "chunks_added": total_chunks
                }
        except asyncio.CancelledError:
            logger.warning(f"Индексация '{filename}' отменена пользователем.")
            self._indexing_progress.pop(filename, None)
            raise
        except Exception as e:
            logger.error(f"Ошибка при индексации '{filename}': {e}")
            self._indexing_progress.pop(filename, None)
            raise
        finally:
            if indexed_any and not success:
                await asyncio.sleep(0.5)
                logger.warning(f"Очистка частичных данных для '{filename}'...")
                try:
                    await asyncio.to_thread(self.delete_document, filename)
                    logger.info(f"Частичные данные для '{filename}' удалены.")
                except Exception as cleanup_err:
                    logger.error(f"Не удалось очистить данные для '{filename}': {cleanup_err}")

    def index_document(self, text: str, filename: str) -> Dict[str, Any]:
        # Синхронная индексация документа
        with self._index_lock:
            logger.info(f"Начало индексации документа: {filename}")

            existing = self.vector_store.get(where={"source": filename}, include=["metadatas"])
            if existing.get("ids"):
                logger.info(f"Документ '{filename}' уже проиндексирован ({len(existing['ids'])} чанков)")
                return {
                    "filename": filename,
                    "chunks_added": 0,
                    "already_indexed": True,
                    "existing_chunks": len(existing["ids"])
                }

            structure_markers = self._parse_structure_markers(text)
            logger.info(f"[{filename}] Структурных маркеров найдено: {len(structure_markers)}")

            doc = Document(page_content=text, metadata={"source": filename})
            chunks = self.text_splitter.split_documents([doc])
            source_version = self._detect_source_version(text)

            for index, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": index,
                    "source_total_chunks": len(chunks),
                    "source_total_chars": len(text),
                    "source_version": source_version,
                })
                if structure_markers:
                    structure = self._get_chunk_structure(
                        structure_markers, chunk.metadata.get("start_index", 0)
                    )
                    chunk.metadata.update(structure)

            for i in range(0, len(chunks), settings.index_batch_size):
                batch = chunks[i:i + settings.index_batch_size]
                self.vector_store.add_documents(batch)
                logger.info(f"[{filename}] Проиндексировано {min(i + settings.index_batch_size, len(chunks))} из {len(chunks)} чанков")

            self._bm25_doc_count = -1
            self._source_chunks_cache = {}
            self._source_chunk_positions = {}

            return {
                "filename": filename,
                "chunks_added": len(chunks),
                "already_indexed": False
            }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        # Удаление книги из базы
        with self._index_lock:
            existing = self.vector_store.get(where={"source": filename}, include=["metadatas"])
            ids = existing.get("ids", [])
            if not ids:
                return {"deleted": False, "filename": filename}
            self.vector_store.delete(ids=ids)
            self._bm25_doc_count = -1
            self._source_chunks_cache = {}
            self._source_chunk_positions = {}
            logger.info(f"Удалено {len(ids)} чанков книги '{filename}'")
            return {"deleted": True, "filename": filename, "chunks_removed": len(ids)}

    def search(self, query: str, top_k: int = 7, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        # Гибридный поиск (векторный + BM25) с ре-ранкингом
        logger.info(f"Поиск: '{query}', top_k={top_k}, sources={sources}")
        
        if sources is not None and len(sources) == 0:
            return []

        reranker = self._load_reranker()
        
        try:
            total_chunks = self.vector_store._collection.count()
        except Exception:
            total_chunks = 0
            
        vector_candidate_k = min(max(top_k * 6, 30), settings.vector_max_candidates)
        if total_chunks > 0:
            bm25_candidate_k = min(
                max(total_chunks // 5, top_k * 20, 100),
                settings.bm25_max_candidates
            )
        else:
            bm25_candidate_k = max(top_k * 20, 100)
            
        queries = [query]
        if settings.query_expansion_enabled and len(query) >= settings.query_expansion_min_length:
            logger.info(f"Выполнение Query Expansion для запроса: '{query}'")
            queries = self._expand_query(query)
            
        query_tokens = self._query_tokens(query)
        ending_intent = self._has_ending_intent(query_tokens, query)

        # Поиск кандидатов через векторное хранилище и BM25
        chroma_filter = {"source": {"$in": sources}} if sources is not None else None
            
        candidate_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        max_start_index = 0

        # Параллельный векторный поиск для всех вариантов запроса
        def _vector_search(q: str):
            return self.vector_store.similarity_search_with_relevance_scores(q, k=vector_candidate_k, filter=chroma_filter)

        with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            vector_results = list(executor.map(_vector_search, queries))

        for v_hits in vector_results:
            for rank, (doc, relevance) in enumerate(v_hits, start=1):
                key = self._doc_key(doc)
                max_start_index = max(max_start_index, key[1])
                candidate = candidate_map.setdefault(
                    key,
                    {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0},
                )
                if candidate["vector_rank"] is None or rank < candidate["vector_rank"]:
                    candidate["vector_rank"] = rank
                    candidate["vector_relevance"] = max(candidate["vector_relevance"], float(relevance))

        # BM25 поиск (индекс пересобирается один раз, затем кэшируется)
        bm25_retriever = self._get_bm25_retriever(k=bm25_candidate_k)
        if bm25_retriever:
            for q in queries:
                b_docs = bm25_retriever.invoke(q)
                if sources is not None:
                    b_docs = [d for d in b_docs if d.metadata.get("source") in sources]
                for rank, doc in enumerate(b_docs, start=1):
                    key = self._doc_key(doc)
                    max_start_index = max(max_start_index, key[1])
                    candidate = candidate_map.setdefault(
                        key,
                        {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0},
                    )
                    if candidate["bm25_rank"] is None or rank < candidate["bm25_rank"]:
                        candidate["bm25_rank"] = rank

        if ending_intent:
            tail_docs = self._get_source_tail_docs()
            if sources is not None:
                tail_docs = [d for d in tail_docs if d.metadata.get("source") in sources]
            for doc in tail_docs:
                key = self._doc_key(doc)
                max_start_index = max(max_start_index, key[1])
                candidate_map.setdefault(
                    key,
                    {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0},
                )

        candidate_docs = [item["doc"] for item in candidate_map.values()]

        if not candidate_docs:
            return []

        # Гибридное ранжирование
        hybrid_ranked: List[Tuple[float, Document]] = []
        for item in candidate_map.values():
            doc = item["doc"]
            vector_rank = item["vector_rank"]
            bm25_rank = item["bm25_rank"]
            lexical_overlap = self._lexical_overlap_score(query_tokens, doc)

            score = 0.0
            if vector_rank is not None:
                score += settings.vector_weight / (vector_rank + 10)
                score += max(0.0, item["vector_relevance"]) * 0.2
            if bm25_rank is not None:
                score += settings.bm25_weight / (bm25_rank + 8)
            score += lexical_overlap * settings.lexical_weight
            if ending_intent:
                score += self._ending_position_boost(doc, max_start_index)

            hybrid_ranked.append((score, doc))

        hybrid_ranked.sort(key=lambda item: item[0], reverse=True)
        
        # Расширение списка кандидатов за счет соседних чанков
        rerank_pool_size = max(vector_candidate_k, top_k * 4)
        expanded_candidates: List[Document] = []
        seen_keys = set()
        for _, doc in hybrid_ranked[: max(top_k * 2, 6)]:
            for candidate in [doc, *self._get_source_neighbors(doc, radius=1)]:
                key = self._doc_key(candidate)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                expanded_candidates.append(candidate)

        if not expanded_candidates:
            expanded_candidates = [doc for _, doc in hybrid_ranked[:rerank_pool_size]]

        rerank_candidates = expanded_candidates[:rerank_pool_size]

        # Финальное ранжирование через Cross-Encoder
        if reranker and len(rerank_candidates) > top_k:
            pairs = [(query, doc.page_content) for doc in rerank_candidates]
            raw_scores = reranker.predict(pairs)
            ranked = sorted(zip(raw_scores, rerank_candidates), key=lambda x: x[0], reverse=True)
            final_docs = [
                (doc, self._reranker_score_to_display(float(score)))
                for score, doc in ranked[:top_k]
            ]
            logger.info(f"Ре-ранкинг: {len(rerank_candidates)} → {top_k} результатов")
        else:
            final_docs = [
                (doc, round(score, 3))
                for score, doc in hybrid_ranked[:top_k]
            ]

        # Форматирование результатов
        MIN_CHUNK_LENGTH = 50
        formatted_results = []
        for doc, score in final_docs:
            if len(doc.page_content.strip()) < MIN_CHUNK_LENGTH:
                continue
            start_idx = doc.metadata.get("start_index", 0)
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Неизвестно"),
                "location": f"Стр. ~{(start_idx // _CHARS_PER_PAGE) + 1} (символ {start_idx})",
                "score": score,
                "tome": doc.metadata.get("tome", ""),
                "part": doc.metadata.get("part", ""),
                "chapter": doc.metadata.get("chapter", ""),
                "epilogue": doc.metadata.get("epilogue", ""),
                "source_version": doc.metadata.get("source_version", ""),
            })

        return formatted_results

    def _build_prompt(self, question: str, search_results: List[Dict[str, Any]]):
        # Подготовка промпта для LLM
        context_notes = self._build_context_notes(question, search_results)
        context_parts = []
        if context_notes:
            context_parts.append("[Служебная заметка]\n" + "\n".join(context_notes) + "\n")
        for i, res in enumerate(search_results):
            structure_parts = [
                v for v in [res.get('tome'), res.get('part'), res.get('chapter'), res.get('epilogue')] if v
            ]
            structure_label = f" | {', '.join(structure_parts)}" if structure_parts else ""
            header = f"[Фрагмент {i+1} из '{res['source']}'{structure_label}]"
            context_parts.append(f"{header}:\n{res['content']}\n")
        return self.qa_prompt.format_messages(
            context="\n".join(context_parts),
            question=question
        )

    async def ask_stream_async(self, question: str, top_k: int = 7, sources: Optional[List[str]] = None) -> AsyncGenerator[str, None]:
        # Потоковая генерация ответа на вопрос
        search_results = await asyncio.to_thread(self.search, question, top_k, sources)

        if not search_results:
            yield json.dumps({"type": "error", "text": "В базе данных пока нет книг. Пожалуйста, загрузите тексты."})
            return

        prompt = self._build_prompt(question, search_results)
        try:
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    yield json.dumps({"type": "chunk", "text": chunk.content})
            yield json.dumps({"type": "sources", "sources": search_results})
        except Exception as e:
            logger.error(f"Ошибка при стриминге от Ollama: {e}")
            yield json.dumps({"type": "error", "text": f"Не удалось получить ответ от модели: {e}"})

    def ask(self, question: str, top_k: int = 7, sources: Optional[List[str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        # Генерация ответа на вопрос
        logger.info(f"Вопрос: '{question}', top_k={top_k}, sources={sources}")

        search_results = self.search(question, top_k=top_k, sources=sources)
        if not search_results:
            return "В базе данных пока нет книг. Пожалуйста, загрузите тексты.", []

        prompt = self._build_prompt(question, search_results)
        try:
            response = self.llm.invoke(prompt)
            return response.content, search_results
        except Exception as e:
            logger.error(f"Ошибка при обращении к Ollama: {e}")
            raise RuntimeError(f"Не удалось получить ответ от модели. Убедитесь, что Ollama запущена. Ошибка: {e}")


rag_service = RAGService()
