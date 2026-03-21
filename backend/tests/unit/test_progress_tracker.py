import threading
import pytest
from unittest.mock import patch


def make_rag_service():
    # Создаёт RAGService с замоканными тяжёлыми зависимостями (torch, chromadb, ollama)
    with (
        patch("services.rag_service._build_embeddings"),
        patch("services.rag_service.Chroma"),
        patch("services.rag_service.ChatOllama"),
        patch("services.rag_service.settings"),
    ):
        from services.rag_service import RAGService
        svc = RAGService.__new__(RAGService)
        svc._indexing_progress = {}
        svc._progress_lock = threading.Lock()
        return svc


class TestProgressTracker:
    def test_set_and_get_progress(self):
        svc = make_rag_service()
        svc._set_progress("book.txt", {"percent": 50, "current": 5, "total": 10})
        result = svc.get_indexing_progress()
        assert result == {"book.txt": {"percent": 50, "current": 5, "total": 10}}

    def test_clear_progress_removes_entry(self):
        svc = make_rag_service()
        svc._set_progress("book.txt", {"percent": 100, "current": 10, "total": 10})
        svc._clear_progress("book.txt")
        assert "book.txt" not in svc.get_indexing_progress()

    def test_clear_progress_missing_key_no_error(self):
        svc = make_rag_service()
        svc._clear_progress("nonexistent.txt")

    def test_get_progress_returns_copy(self):
        # Мутация возвращённого dict не должна влиять на внутреннее состояние
        svc = make_rag_service()
        svc._set_progress("book.txt", {"percent": 0, "current": 0, "total": 10})
        snapshot = svc.get_indexing_progress()
        snapshot["book.txt"]["percent"] = 999
        assert svc.get_indexing_progress()["book.txt"]["percent"] == 0

    def test_concurrent_set_progress_no_data_race(self):
        # 10 потоков пишут одновременно — не должно быть искажения состояния
        svc = make_rag_service()
        errors = []

        def write_progress(book_id: int):
            try:
                for i in range(20):
                    svc._set_progress(f"book_{book_id}.txt", {"percent": i * 5})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_progress, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(svc.get_indexing_progress()) == 10
