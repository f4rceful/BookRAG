from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from api.dependencies import limiter
from api.routes import router
from config import settings

app = FastAPI(
    title="BookRAG API",
    description="API для умного поиска по книгам с использованием локальных моделей",
    version="1.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _auto_index_books(books_dir: str, logger) -> None:
    import os
    import asyncio
    import chardet
    from services.rag_service import rag_service

    txt_files = sorted(f for f in os.listdir(books_dir) if f.lower().endswith(".txt"))
    if not txt_files:
        return

    for filename in txt_files:
        existing = await asyncio.to_thread(
            rag_service.vector_store.get,
            where={"source": filename},
            include=["metadatas"]
        )
        existing_ids = existing.get("ids", [])
        existing_metas = existing.get("metadatas", [])
        if existing_ids:
            expected = existing_metas[0].get("source_total_chunks", 0) if existing_metas else 0
            actual = len(existing_ids)
            if expected > 0 and actual >= expected:
                logger.info(f"📚 Книга уже в базе, пропускаем: {filename} ({actual} чанков)")
                continue
            logger.warning(f"⚠️ Частичная индексация '{filename}': {actual}/{expected} чанков. Переиндексируем...")

        filepath = os.path.join(books_dir, filename)
        logger.info(f"📖 Авто-индексация: {filename}")
        try:
            with open(filepath, "rb") as f:
                content = f.read()

            # Определяем кодировку через chardet, затем пробуем UTF-8 и CP1251
            detected = chardet.detect(content)
            detected_enc = detected.get("encoding") or "utf-8"
            text = None
            for enc in [detected_enc, "utf-8", "cp1251"]:
                try:
                    text = content.decode(enc)
                    logger.info(f"Файл {filename} прочитан в кодировке {enc}")
                    break
                except (UnicodeDecodeError, LookupError):
                    continue

            if text is None:
                text = content.decode("utf-8", errors="replace")
                logger.warning(f"Файл {filename}: кодировка не определена, используем UTF-8 с заменой")

            async for step in rag_service.index_document_async(text, filename):
                if step.get("type") == "success":
                    logger.info(f"✅ Проиндексировано: {filename} ({step.get('chunks_added', '?')} чанков)")
                elif step.get("type") == "error":
                    logger.error(f"❌ Ошибка индексации {filename}: {step.get('detail')}")
        except Exception as e:
            logger.error(f"❌ Не удалось проиндексировать {filename}: {e}")


@app.on_event("startup")
async def startup_event():
    import os
    import asyncio
    import chardet
    import httpx
    import logging
    logger = logging.getLogger("uvicorn")

    # Проверка соединения с Ollama
    logger.info("Проверка связи с Ollama...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if settings.model_name not in models:
                    logger.warning(f"⚠️ Модель '{settings.model_name}' не найдена в Ollama. "
                                   f"Пожалуйста, выполните 'ollama pull {settings.model_name}'")
                else:
                    logger.info(f"✅ Модель '{settings.model_name}' готова к работе.")
    except Exception as e:
        logger.error(f"❌ Не удалось подключиться к Ollama на {settings.ollama_base_url}: {e}")

    # Фоновая индексация книг из директории books
    books_dir = os.path.abspath(settings.books_directory)
    os.makedirs(books_dir, exist_ok=True)
    asyncio.create_task(_auto_index_books(books_dir, logger))

app.include_router(router, prefix="/api")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "BookRAG Backend is running"}
