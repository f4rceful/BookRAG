import asyncio
import os
import json
import httpx
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from api.schemas import SearchResponse, AskResponse
from api.dependencies import limiter
from services.rag_service import rag_service
from services.book_parser import detect_and_decode

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    sources: Optional[List[str]] = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    sources: Optional[List[str]] = None

class ModelSetRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)

@router.get("/models", summary="Получить текущую модель")
def get_current_model():
    return {"current_model": rag_service.get_current_model()}

@router.get("/models/available", summary="Список моделей доступных в Ollama")
def get_available_models():
    try:
        ollama_url = rag_service.llm.base_url.rstrip("/")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"models": models, "current": rag_service.get_current_model()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama недоступна: {str(e)}")

@router.post("/model", summary="Изменить модель Ollama")
def set_model(request: ModelSetRequest):
    try:
        ollama_url = rag_service.llm.base_url.rstrip("/")
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        if request.model_name not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Модель '{request.model_name}' не найдена в Ollama. Доступные: {available}"
            )
        result = rag_service.set_model(request.model_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при смене модели: {str(e)}")

@router.get("/indexing-progress", summary="Прогресс авто-индексации книг")
def get_indexing_progress():
    return {"progress": rag_service.get_indexing_progress()}

@router.get("/books", summary="Получить список загруженных книг")
async def get_books():
    try:
        books = await asyncio.to_thread(rag_service.get_books)
        return {"books": books, "total": len(books)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@router.get("/stats", summary="Общая статистика базы")
async def get_stats():
    try:
        books = await asyncio.to_thread(rag_service.get_books)
        count = await asyncio.to_thread(rag_service.vector_store._collection.count)
        return {"books_count": len(books), "chunks_count": count}
    except Exception as e:
        return {"books_count": 0, "chunks_count": 0, "error": str(e)}

@router.delete("/book/{filename}", summary="Удалить книгу из базы")
async def delete_book(filename: str):
    try:
        result = await asyncio.to_thread(rag_service.delete_document, filename)
        if not result["deleted"]:
            raise HTTPException(status_code=404, detail=f"Книга '{filename}' не найдена в базе.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении: {str(e)}")

@router.post("/upload", summary="Загрузить книгу")
async def upload_book(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename or "").strip()
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Некорректное имя файла.")
    
    if not safe_filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Поддерживаются только .txt файлы")

    try:
        content = await file.read()

        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Файл слишком большой. Максимальный размер: {MAX_UPLOAD_SIZE // 1024 // 1024} МБ."
            )

        text = detect_and_decode(content, filename=safe_filename)

        async def generate_progress():
            try:
                async for step in rag_service.index_document_async(text, safe_filename):
                    yield f"data: {json.dumps(step)}\n\n"
            except asyncio.CancelledError:
                logger.warning(f"Индексация '{safe_filename}' прервана клиентом.")
                # очистка частичных данных происходит в finally блоке rag_service
                raise
            except Exception as e:
                logger.error(f"Ошибка при индексации: {e}")
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

        return StreamingResponse(generate_progress(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

async def _validate_sources(sources: Optional[List[str]]) -> None:
    # Проверяет что все запрошенные источники существуют в базе, иначе 400
    if not sources:
        return
    books = await asyncio.to_thread(rag_service.get_books)
    known = set(books)
    invalid = [s for s in sources if s not in known]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Книги не найдены в базе: {invalid}. Доступные: {sorted(known)}"
        )


@router.post("/search", response_model=SearchResponse, summary="Поиск фрагментов по книгам")
async def search(request: SearchRequest):
    await _validate_sources(request.sources)
    results = await asyncio.to_thread(rag_service.search, request.query, request.top_k, request.sources)
    return {"results": results}

@router.post("/ask", response_model=AskResponse, summary="Вопрос по тексту книг")
@limiter.limit("10/minute")
async def ask_question(request: Request, body: AskRequest):
    await _validate_sources(body.sources)
    try:
        answer, found_sources = await asyncio.to_thread(rag_service.ask, body.question, body.top_k, body.sources)
        return {"answer": answer, "sources": found_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации ответа: {str(e)}")

@router.post("/ask/stream", summary="Стриминг ответа по тексту книг (SSE)")
@limiter.limit("10/minute")
async def ask_stream(request: Request, body: AskRequest):
    await _validate_sources(body.sources)

    async def generate():
        try:
            async for chunk in rag_service.ask_stream_async(body.question, body.top_k, body.sources):
                yield f"data: {chunk}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")
