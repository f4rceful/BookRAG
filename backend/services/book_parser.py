import logging
from pathlib import Path
from typing import Union

import chardet

logger = logging.getLogger(__name__)


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
