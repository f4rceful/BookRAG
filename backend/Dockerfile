FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директорию для ChromaDB
RUN mkdir -p chroma_db

EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
