import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import settings

logger = logging.getLogger(__name__)

# Примерное количество символов на одну страницу книги (для отображения локации)
_CHARS_PER_PAGE = 1800

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


class SearchService:
    def __init__(self, vector_store: Chroma, get_llm: Callable):
        # get_llm — callable возвращающий текущий ChatOllama (для query expansion)
        self._vector_store = vector_store
        self._get_llm = get_llm

        self._query_expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помогаешь улучшить поиск по книгам.
Переформулируй вопрос пользователя 2 разными способами — короткими конкретными фразами,
которые могут встретиться в художественном тексте.
Верни ТОЛЬКО 2 строки без нумерации и объяснений."""),
            ("user", "{question}")
        ])

        self._bm25_preprocess_func = self._build_bm25_preprocess_func()
        self._bm25_dirty: bool = True
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._bm25_lock = threading.Lock()

        self._source_chunks_cache: Dict[str, List[Document]] = {}
        self._source_chunk_positions: Dict[str, Dict[int, int]] = {}

        self._reranker_tried: bool = False
        self._reranker = None

    # --- Управление кэшем ---

    def invalidate_cache(self) -> None:
        # Вызывается фасадом после записи/удаления документа
        with self._bm25_lock:
            self._bm25_dirty = True
            self._source_chunks_cache = {}
            self._source_chunk_positions = {}

    # --- BM25 ---

    @staticmethod
    def _basic_bm25_tokens(text: str) -> List[str]:
        return [
            token.lower()
            for token in _TOKEN_RE.findall(text)
            if len(token) > 2 and token.lower() not in _STOPWORDS
        ]

    @staticmethod
    def _fallback_russian_stem(token: str) -> str:
        # Упрощённый стемминг: обрезает типичные русские окончания если NLTK недоступен
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
        # Строит препроцессор токенов для BM25: SnowballStemmer если NLTK доступен, иначе fallback
        try:
            from nltk.stem.snowball import SnowballStemmer
            stemmer = SnowballStemmer("russian")
            logger.info("BM25 использует SnowballStemmer для русского языка")

            def preprocess(text: str) -> List[str]:
                return [stemmer.stem(token) for token in self._basic_bm25_tokens(text)]

            return preprocess
        except Exception as exc:
            logger.warning("NLTK недоступен, BM25 использует встроенный русский стеммер: %s", exc)

            def preprocess(text: str) -> List[str]:
                return [self._fallback_russian_stem(token) for token in self._basic_bm25_tokens(text)]

            return preprocess

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
                int(doc.metadata.get("start_index", 0)): idx
                for idx, doc in enumerate(sorted_docs)
            }

    def _get_bm25_retriever(self, k: int) -> Optional[BM25Retriever]:
        # Dirty flag: перестраиваем индекс только после записи/удаления, не на каждый поиск
        with self._bm25_lock:
            if self._bm25_dirty:
                try:
                    data = self._vector_store.get(include=["documents", "metadatas"])
                except Exception:
                    return None
                if not data.get("documents"):
                    return None
                logger.info(f"Пересборка BM25-индекса ({len(data['documents'])} чанков)...")
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(data["documents"], data["metadatas"])
                ]
                self._rebuild_source_cache(docs)
                self._bm25_retriever = BM25Retriever.from_documents(
                    docs, preprocess_func=self._bm25_preprocess_func
                )
                self._bm25_dirty = False
                logger.info("BM25-индекс готов")

            if self._bm25_retriever is None:
                return None
            self._bm25_retriever.k = k
            return self._bm25_retriever

    # --- Вспомогательные методы поиска ---

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

    @staticmethod
    def _has_ending_intent(query_tokens: List[str], raw_query: str) -> bool:
        raw_lower = raw_query.lower()
        if any(term in raw_lower for term in _ENDING_QUERY_TERMS):
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
    def _doc_key(doc: Document) -> Tuple[str, int]:
        return doc.metadata.get("source", ""), int(doc.metadata.get("start_index", 0))

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

    def _load_reranker(self):
        # Ленивая загрузка Cross-Encoder при первом обращении
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
        # Генерирует 2 переформулировки вопроса через LLM для улучшения полноты поиска
        try:
            llm = self._get_llm()
            resp = llm.invoke(self._query_expansion_prompt.format_messages(question=question))
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
        # Нормализует сырой logit Cross-Encoder в диапазон [0, 1] через сигмоиду
        return round(1.0 / (1.0 + math.exp(-raw_score / 3.0)), 3)

    # --- Основной поиск ---

    def search(
        self, query: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # Гибридный поиск: векторный (cosine similarity) + BM25 (лексический) + lexical overlap boost
        logger.info(f"Поиск: '{query}', top_k={top_k}, sources={sources}")

        if sources is not None and len(sources) == 0:
            return []

        self._load_reranker()

        try:
            total_chunks = self._vector_store._collection.count()
        except Exception:
            total_chunks = 0

        vector_candidate_k = min(max(top_k * 6, 30), settings.vector_max_candidates)
        if total_chunks > 0:
            bm25_candidate_k = min(
                max(total_chunks // 5, top_k * 20, 100),
                settings.bm25_max_candidates,
            )
        else:
            bm25_candidate_k = max(top_k * 20, 100)

        queries = [query]
        if settings.query_expansion_enabled and len(query) >= settings.query_expansion_min_length:
            logger.info(f"Выполнение Query Expansion для запроса: '{query}'")
            queries = self._expand_query(query)

        query_tokens = self._query_tokens(query)
        ending_intent = self._has_ending_intent(query_tokens, query)

        # Фильтр по источникам передаётся напрямую в ChromaDB
        chroma_filter = {"source": {"$in": sources}} if sources is not None else None

        candidate_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        max_start_index = 0

        # Параллельный векторный поиск для всех вариантов запроса
        def _vector_search(q: str):
            return self._vector_store.similarity_search_with_relevance_scores(
                q, k=vector_candidate_k, filter=chroma_filter
            )

        with ThreadPoolExecutor(max_workers=min(len(queries), 3)) as executor:
            vector_results = list(executor.map(_vector_search, queries))

        for v_hits in vector_results:
            for rank, (doc, relevance) in enumerate(v_hits, start=1):
                key = self._doc_key(doc)
                max_start_index = max(max_start_index, key[1])
                candidate = candidate_map.setdefault(
                    key, {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0}
                )
                if candidate["vector_rank"] is None or rank < candidate["vector_rank"]:
                    candidate["vector_rank"] = rank
                    candidate["vector_relevance"] = max(candidate["vector_relevance"], float(relevance))

        # BM25 фильтрует по источникам вручную, так как не поддерживает фильтры ChromaDB
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
                        key, {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0}
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
                    key, {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_relevance": 0.0}
                )

        if not candidate_map:
            return []

        # Reciprocal Rank Fusion: score = vector_w/(rank+10) + bm25_w/(rank+8) + lexical_w*overlap
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

        hybrid_ranked.sort(key=lambda x: x[0], reverse=True)

        # Добавляем соседние чанки чтобы не обрывать контекст на границах сплита
        # radius=2: покрываем до ~1600 символов контекста вокруг каждого найденного чанка
        score_map = {self._doc_key(doc): score for score, doc in hybrid_ranked}
        expanded_scored: List[Tuple[float, Document]] = []
        seen_keys: set = set()
        for score, doc in hybrid_ranked[: max(top_k * 2, 6)]:
            for neighbor in self._get_source_neighbors(doc, radius=2):
                key = self._doc_key(neighbor)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Соседние чанки получают 70% скора родителя — приоритет у прямых попаданий
                neighbor_score = score_map.get(key, score * 0.7)
                expanded_scored.append((neighbor_score, neighbor))

            key = self._doc_key(doc)
            if key not in seen_keys:
                seen_keys.add(key)
                expanded_scored.append((score, doc))

        if not expanded_scored:
            expanded_scored = list(hybrid_ranked[:top_k])

        expanded_scored.sort(key=lambda x: x[0], reverse=True)

        # Cross-Encoder реранкинг отключён — слишком медленный на CPU без GPU буста
        final_docs = [(doc, round(score, 3)) for score, doc in expanded_scored[:top_k]]
        logger.info(f"Используется базовое ранжирование с expansion(radius=2): {top_k} результатов")

        # Фильтруем слишком короткие результаты и добавляем метаданные позиции
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
