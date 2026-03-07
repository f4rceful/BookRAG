FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Собираем React для продакшена (опционально можно оставить dev server, но лучше preview)
RUN npm run build

# Устанавливаем легкий сервер для статики
RUN npm install -g serve

# Запускаем
CMD ["serve", "-s", "dist", "-l", "5173"]