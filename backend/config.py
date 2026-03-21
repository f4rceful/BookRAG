from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8"
    )

    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b"
    embedding_model: str = "BAAI/bge-m3"
    chroma_persist_directory: str = str(_BASE_DIR / "chroma_db")
    books_directory: str = str(_BASE_DIR / "books")

    # Параметры разбиения текста на фрагменты
    chunk_size: int = 800
    chunk_overlap: int = 300
    index_batch_size: int = 100

    # Лимиты кандидатов для гибридного поиска
    bm25_max_candidates: int = 500
    vector_max_candidates: int = 60

    # Расширение запроса через LLM (отключено для скорости)
    query_expansion_enabled: bool = False
    query_expansion_min_length: int = 20

    # Веса гибридного поиска
    vector_weight: float = 0.5
    bm25_weight: float = 0.6
    lexical_weight: float = 1.8

    # Настройки безопасности и документации
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    docs_enabled: bool = True

settings = Settings()
