import logging
import re
from pathlib import Path
from typing import Union

import chardet

logger = logging.getLogger(__name__)

# Шаблон для поиска структурных заголовков книги (Том, Часть, Глава, Эпилог)
_STRUCTURE_RE = re.compile(
    r'(?P<tome>^(?:ТОМ|Том)\s+\S+)'
    r'|(?P<part>^(?:ЧАСТЬ|Часть)\s+\S+)'
    r'|(?P<chapter>^(?:ГЛАВА|Глава)\s+\S+)'
    r'|(?P<epilogue>^(?:ЭПИЛОГ|Эпилог)\s*)',
    re.MULTILINE | re.UNICODE
)


def detect_and_decode(content: bytes, filename: str = "<unknown>") -> str:
    # Определяет кодировку файла и декодирует байты в строку.
    # Порядок попыток:
    #   1. Строгий UTF-8 (большинство современных файлов)
    #   2. Кодировка определённая через chardet
    #   3. windows-1251 / cp1251 (типично для русской литературы)
    #   4. latin-1 (последний вариант без потерь для однобайтных кодировок)
    #   5. UTF-8 с заменой символов (гарантированный fallback)

    # Быстрый путь: сначала пробуем строгий UTF-8
    try:
        text = content.decode("utf-8")
        logger.info(f"[{filename}] Кодировка: UTF-8 (строгий)")
        return text
    except UnicodeDecodeError:
        pass

    # Пробуем chardet и распространённые кириллические кодировки
    detected = chardet.detect(content)
    detected_enc = detected.get("encoding") or ""

    for enc in [detected_enc, "windows-1251", "cp1251", "latin-1"]:
        if not enc:
            continue
        try:
            text = content.decode(enc)
            logger.info(f"[{filename}] Кодировка: {enc}")
            return text
        except (UnicodeDecodeError, LookupError):
            continue

    # Гарантированный fallback: UTF-8 с заменой нераспознанных символов
    text = content.decode("utf-8", errors="replace")
    logger.warning(f"[{filename}] Кодировка не определена — UTF-8 с заменой символов")
    return text


def read_book_file(path: Union[str, Path]) -> str:
    # Читает .txt файл книги с диска и возвращает текст
    path = Path(path)
    content = path.read_bytes()
    return detect_and_decode(content, filename=path.name)


def parse_structure_markers(text: str) -> list:
    # Ищет заголовки (Том, Часть, Глава, Эпилог) в тексте книги
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


def get_chunk_structure(markers: list, chunk_start: int) -> dict:
    # Определяет структуру (Том/Часть/Глава/Эпилог) для фрагмента по его позиции в тексте
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


def detect_source_version(text: str) -> str:
    # Проверяет первые 2000 символов — определяет является ли текст черновиком
    head = text[:2000].lower()
    if "первый вариант романа" in head:
        return "first_draft"
    return "standard"
