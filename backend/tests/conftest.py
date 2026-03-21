import pytest


@pytest.fixture
def sample_russian_utf8() -> bytes:
    # UTF-8 encoded Russian text
    return "Война и мир. Лев Толстой.".encode("utf-8")


@pytest.fixture
def sample_russian_cp1251() -> bytes:
    # Windows-1251 encoded Russian text (типично для старых русских книг)
    return "Война и мир. Лев Толстой.".encode("windows-1251")


@pytest.fixture
def sample_book_text() -> str:
    # Минимальный текст для тестов индексации и поиска
    return """\
Часть первая

Глава I

Анна Павловна Шерер, фрейлина и приближённая императрицы Марии Феодоровны,
встречала гостей в своей петербургской гостиной.

Глава II

Князь Андрей Болконский приехал в Москву вечером.
Он был утомлён долгой дорогой, но рад возвращению домой.
"""
