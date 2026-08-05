

FROM python:3.11-slim

WORKDIR /app

# Crucial: Ensures 'from src.xxx import yyy' works inside Docker
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}"]