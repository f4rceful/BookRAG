import pytest
from services.book_parser import detect_and_decode


class TestDetectAndDecode:
    def test_utf8_decoded_correctly(self, sample_russian_utf8):
        result = detect_and_decode(sample_russian_utf8, "test.txt")
        assert result == "Война и мир. Лев Толстой."

    def test_cp1251_decoded_correctly(self, sample_russian_cp1251):
        result = detect_and_decode(sample_russian_cp1251, "test.txt")
        assert result == "Война и мир. Лев Толстой."

    def test_latin_ascii_decoded_correctly(self):
        result = detect_and_decode(b"Hello, world!", "ascii.txt")
        assert result == "Hello, world!"

    def test_empty_bytes_returns_empty_string(self):
        result = detect_and_decode(b"", "empty.txt")
        assert result == ""

    def test_fallback_does_not_raise(self):
        # Мусорные байты которые chardet не может распознать — должна вернуться строка без ошибки
        garbage = bytes(range(128, 256))
        result = detect_and_decode(garbage, "garbage.txt")
        assert isinstance(result, str)

    def test_utf8_bom_handled(self):
        # UTF-8 BOM (0xEF BB BF) встречается в файлах сохранённых через Notepad
        text = "Привет мир"
        content = b"\xef\xbb\xbf" + text.encode("utf-8")
        result = detect_and_decode(content, "bom.txt")
        assert "Привет мир" in result
