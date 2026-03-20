FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для сборки некоторых пакетов (например, для chromadb или sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Сначала устанавливаем torch, чтобы pip не пытался выкачать его как зависимость в неправильной версии
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Копируем и устанавливаем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем необходимые ресурсы для nltk (если понадобятся стеммеры)
RUN python -m nltk.downloader punkt_tab

# Копируем весь код
COPY . .

# Создаем директории для данных
RUN mkdir -p chroma_db books

EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
