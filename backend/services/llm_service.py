import asyncio
import json
import logging
import threading
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import settings
from services.searcher import SearchService

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, get_llm: Callable[[], ChatOllama], searcher: SearchService):
        # get_llm — callable возвращающий текущий ChatOllama под model_lock
        self._get_llm = get_llm
        self._searcher = searcher

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """ВАЖНО: Отвечай ИСКЛЮЧИТЕЛЬНО на русском языке. Никогда не используй китайский, английский или любой другой язык. Только русский.

Ты — умный литературный ассистент. Используй предоставленные фрагменты книги как основу для своего ответа.

Тебе разрешено:
- Анализировать и сопоставлять информацию из разных фрагментов
- Делать логические выводы из нескольких фрагментов — например, если в одном фрагменте "Онегин выстрелил", а в другом "Ленский убит", ты ОБЯЗАН сделать вывод: Онегин убил Ленского
- Описывать мотивы персонажей и причинно-следственные связи на основе фрагментов
- Синтезировать целостный ответ, объединяя несколько фрагментов
- Давать уверенный ответ когда логический вывод очевиден из совокупности фрагментов

Тебе запрещено:
- Выдумывать факты, которых нет ни в одном из фрагментов
- Добавлять внешние знания, не относящиеся к тексту книги
- Говорить "неизвестно" или "нельзя определить" если ответ логически следует из имеющихся фрагментов
- Использовать любой язык кроме русского

Если у фрагментов есть метки Том/Часть/Глава/Эпилог — используй их для понимания хронологии событий.
Если фрагменты не содержат достаточно информации для ответа, скажи: «К сожалению, в загруженных материалах недостаточно информации для ответа на этот вопрос.»

Фрагменты текста (Контекст):
{context}

Напоминание: твой ответ должен быть написан ТОЛЬКО на русском языке."""),
            ("user", "Ответь на русском языке: {question}")
        ])

    # --- Подготовка промпта ---

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

    def _build_prompt(self, question: str, search_results: List[Dict[str, Any]]):
        context_notes = self._build_context_notes(question, search_results)
        context_parts = []
        if context_notes:
            context_parts.append("[Служебная заметка]\n" + "\n".join(context_notes) + "\n")
        for i, res in enumerate(search_results):
            structure_parts = [
                v for v in [res.get('tome'), res.get('part'), res.get('chapter'), res.get('epilogue')] if v
            ]
            structure_label = f" | {', '.join(structure_parts)}" if structure_parts else ""
            header = f"[Фрагмент {i + 1} из '{res['source']}'{structure_label}]"
            context_parts.append(f"{header}:\n{res['content']}\n")
        return self.qa_prompt.format_messages(
            context="\n".join(context_parts),
            question=question,
        )

    # --- Ответы ---

    def ask(
        self, question: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info(f"Вопрос: '{question}', top_k={top_k}, sources={sources}")
        search_results = self._searcher.search(question, top_k=top_k, sources=sources)
        if not search_results:
            return "В базе данных пока нет книг. Пожалуйста, загрузите тексты.", []
        prompt = self._build_prompt(question, search_results)
        try:
            response = self._get_llm().invoke(prompt)
            return response.content, search_results
        except Exception as e:
            logger.error(f"Ошибка при обращении к Ollama: {e}")
            raise RuntimeError(
                f"Не удалось получить ответ от модели. Убедитесь, что Ollama запущена. Ошибка: {e}"
            )

    async def ask_stream_async(
        self, question: str, top_k: int = 7, sources: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        search_results = await asyncio.to_thread(
            self._searcher.search, question, top_k, sources
        )
        if not search_results:
            yield json.dumps({"type": "error", "text": "В базе данных пока нет книг. Пожалуйста, загрузите тексты."})
            return
        prompt = self._build_prompt(question, search_results)
        try:
            async for chunk in self._get_llm().astream(prompt):
                if chunk.content:
                    yield json.dumps({"type": "chunk", "text": chunk.content})
            yield json.dumps({"type": "sources", "sources": search_results})
        except Exception as e:
            logger.error(f"Ошибка при стриминге от Ollama: {e}")
            yield json.dumps({"type": "error", "text": f"Не удалось получить ответ от модели: {e}"})
