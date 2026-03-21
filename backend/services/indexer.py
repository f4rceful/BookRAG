import asyncio
import copy
import logging
import threading
from typing import Any, AsyncGenerator, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from services.book_parser import detect_source_version, get_chunk_structure, parse_structure_markers

logger = logging.getLogger(__name__)

# Короткие фрагменты (< 100 символов) не несут смысловой нагрузки
_MIN_CHUNK_LENGTH = 100


class IndexerService:
    def __init__(self, vector_store: Chroma, text_splitter: RecursiveCharacterTextSplitter):
        self._vector_store = vector_store
        self._text_splitter = text_splitter
        self._indexing_progress: Dict[str, Dict] = {}
        self._progress_lock = threading.Lock()
        self._index_lock = threading.Lock()

    # --- Книги ---

    def get_books(self) -> List[str]:
        results = self._vector_store.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sorted(list(sources))

    def count_chunks(self) -> int:
        try:
            return self._vector_store._collection.count()
        except Exception:
            return 0

    def check_book_status(self, filename: str) -> Dict[str, Any]:
        # Возвращает сколько чанков есть и сколько ожидается (для проверки неполной индексации)
        existing = self._vector_store.get(where={"source": filename}, include=["metadatas"])
        ids = existing.get("ids", [])
        metas = existing.get("metadatas", [])
        expected = metas[0].get("source_total_chunks", 0) if metas else 0
        return {"ids": ids, "actual": len(ids), "expected": expected}

    def delete_document(self, filename: str) -> Dict[str, Any]:
        with self._index_lock:
            existing = self._vector_store.get(where={"source": filename}, include=["metadatas"])
            ids = existing.get("ids", [])
            if not ids:
                return {"deleted": False, "filename": filename}
            self._vector_store.delete(ids=ids)
            logger.info(f"Удалено {len(ids)} чанков книги '{filename}'")
            return {"deleted": True, "filename": filename, "chunks_removed": len(ids)}

    # --- Прогресс ---

    def get_indexing_progress(self) -> Dict[str, Dict]:
        # Возвращает глубокую копию чтобы внешний код не мог изменить внутреннее состояние
        with self._progress_lock:
            return copy.deepcopy(self._indexing_progress)

    def _set_progress(self, filename: str, data: Dict) -> None:
        with self._progress_lock:
            self._indexing_progress[filename] = data

    def _clear_progress(self, filename: str) -> None:
        with self._progress_lock:
            self._indexing_progress.pop(filename, None)

    # --- Внутренняя логика ---

    def _build_chunks(self, text: str, filename: str) -> List[Document]:
        # Нарезает текст и обогащает метаданные каждого фрагмента структурой книги
        structure_markers = parse_structure_markers(text)
        source_version = detect_source_version(text)
        doc = Document(page_content=text, metadata={"source": filename})
        chunks = self._text_splitter.split_documents([doc])
        chunks = [c for c in chunks if len(c.page_content.strip()) >= _MIN_CHUNK_LENGTH]
        total = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": index,
                "source_total_chunks": total,
                "source_total_chars": len(text),
                "source_version": source_version,
            })
            if structure_markers:
                structure = get_chunk_structure(
                    structure_markers, chunk.metadata.get("start_index", 0)
                )
                chunk.metadata.update(structure)
        return chunks

    # --- Индексация ---

    def index_document(self, text: str, filename: str) -> Dict[str, Any]:
        with self._index_lock:
            logger.info(f"Начало индексации документа: {filename}")
            existing = self._vector_store.get(where={"source": filename}, include=["metadatas"])
            if existing.get("ids"):
                logger.info(f"Документ '{filename}' уже проиндексирован ({len(existing['ids'])} чанков)")
                return {
                    "filename": filename,
                    "chunks_added": 0,
                    "already_indexed": True,
                    "existing_chunks": len(existing["ids"]),
                }
            chunks = self._build_chunks(text, filename)
            for i in range(0, len(chunks), settings.index_batch_size):
                batch = chunks[i:i + settings.index_batch_size]
                self._vector_store.add_documents(batch)
                logger.info(
                    f"[{filename}] Проиндексировано "
                    f"{min(i + settings.index_batch_size, len(chunks))} из {len(chunks)} чанков"
                )
            return {"filename": filename, "chunks_added": len(chunks), "already_indexed": False}

    async def index_document_async(
        self, text: str, filename: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # Индексирует документ батчами, транслируя прогресс через SSE.
        # Если индексация прервана или упала — откатывает частичные данные.
        indexed_any = False
        success = False

        # Удаляем старые чанки если предыдущая индексация была прервана
        try:
            existing = await asyncio.to_thread(
                self._vector_store.get,
                where={"source": filename},
                include=["metadatas"],
            )
            if existing.get("ids"):
                logger.info(f"Обнаружены частичные данные для '{filename}'. Очистка...")
                await asyncio.to_thread(self.delete_document, filename)
        except Exception as e:
            logger.warning(f"Ошибка при предварительной очистке '{filename}': {e}")

        try:
            with self._index_lock:
                logger.info(f"Начало индексации документа: {filename}")
                chunks = await asyncio.to_thread(self._build_chunks, text, filename)
                total_chunks = len(chunks)
                self._set_progress(filename, {"percent": 0, "current": 0, "total": total_chunks})
                yield {"type": "start", "filename": filename, "total_chunks": total_chunks}

                for i in range(0, total_chunks, settings.index_batch_size):
                    batch = chunks[i:i + settings.index_batch_size]
                    await asyncio.to_thread(self._vector_store.add_documents, batch)
                    indexed_any = True
                    current_count = min(i + settings.index_batch_size, total_chunks)
                    pct = round((current_count / total_chunks) * 100)
                    self._set_progress(
                        filename, {"percent": pct, "current": current_count, "total": total_chunks}
                    )
                    yield {"type": "progress", "current": current_count, "total": total_chunks, "percent": pct}
                    await asyncio.sleep(0.05)

                success = True
                self._clear_progress(filename)
                yield {"type": "success", "filename": filename, "chunks_added": total_chunks}

        except asyncio.CancelledError:
            logger.warning(f"Индексация '{filename}' отменена пользователем.")
            self._clear_progress(filename)
            raise
        except Exception as e:
            logger.error(f"Ошибка при индексации '{filename}': {e}")
            self._clear_progress(filename)
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
