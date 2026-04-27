FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY main.py .

# Порт, который будет слушать сервер
EXPOSE 8000

# Переменная окружения по умолчанию (реальный ключ передаётся при запуске)
ENV OPENROUTER_API_KEY=""

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
