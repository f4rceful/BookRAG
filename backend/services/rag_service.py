import logging
import os
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from services.embeddings import build_embeddings
from services.indexer import IndexerService
from services.llm_service import LLMService
from services.searcher import SearchService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class RAGService:
    # Фасад: создаёт и компонует сервисы, предоставляет единый публичный API для routes.py
    def __init__(self):
        logger.info(f"Инициализация эмбеддингов: {settings.embedding_model}")
        embeddings = build_embeddings(settings.embedding_model)

        persist_dir = os.path.abspath(settings.chroma_persist_directory)
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"Инициализация ChromaDB: {persist_dir}")
        vector_store = Chroma(
            collection_name="books_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Инициализация Ollama: {settings.model_name} на {settings.ollama_base_url}")
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.model_name,
            temperature=0.1,
            num_ctx=3072,
            timeout=180,  # Даем Ollama 3 минуты, чтобы она успела все обдумать на AMD
        )
        self._model_lock = threading.Lock()
        self._current_model_name = settings.model_name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )

        self._indexer = IndexerService(vector_store, text_splitter)
        self._searcher = SearchService(vector_store, get_llm=self._get_llm)
        self._llm_svc = LLMService(get_llm=self._get_llm, searcher=self._searcher)

    def _get_llm(self) -> ChatOllama:
        # Возвращает текущий LLM под блокировкой — защита от гонки при смене модели
        with self._model_lock:
            return self._llm

    # --- Управление моделью ---

    def get_ollama_url(self) -> str:
        return self._llm.base_url.rstrip("/")

    def get_current_model(self) -> str:
        return self._current_model_name

    def set_model(self, model_name: str) -> Dict[str, str]:
        with self._model_lock:
            logger.info(f"Смена модели на: {model_name}")
            self._llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=model_name,
                temperature=0.1,
            )
            self._current_model_name = model_name
        return {"status": "success", "model": model_name}

    # --- Книги и статистика ---

    def get_books(self) -> List[str]:
        return self._indexer.get_books()

    def count_chunks(self) -> int:
        return self._indexer.count_chunks()

    def check_book_status(self, filename: str) -> Dict[str, Any]:
        return self._indexer.check_book_status(filename)

    def get_indexing_progress(self) -> Dict[str, Dict]:
        return self._indexer.get_indexing_progress()

    # --- Индексация ---

    def delete_document(self, filename: str) -> Dict[str, Any]:
        result = self._indexer.delete_document(filename)
        if result["deleted"]:
            self._searcher.invalidate_cache()
        return result

    def index_document(self, text: str, filename: str) -> Dict[str, Any]:
        result = self._indexer.index_document(text, filename)
        if not result.get("already_indexed"):
            self._searcher.invalidate_cache()
        return result

    async def index_document_async(
        self, text: str, filename: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # После успешной индексации инвалидируем BM25 кэш
        async for step in self._indexer.index_document_async(text, filename):
            if step.get("type") == "success":
                self._searcher.invalidate_cache()
            yield step

    # --- Поиск и ответы ---

    def search(
        self, query: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        return self._searcher.search(query, top_k, sources)

    def ask(
        self, question: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        return self._llm_svc.ask(question, top_k, sources)

    async def ask_stream_async(
        self, question: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._llm_svc.ask_stream_async(question, top_k, sources):
            yield chunk


rag_service = RAGService()
