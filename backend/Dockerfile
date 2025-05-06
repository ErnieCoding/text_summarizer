FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir flask redis celery gunicorn flask-cors

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "1200", "main:app"]
