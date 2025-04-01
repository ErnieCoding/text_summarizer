
FROM python:3.10-slim
WORKDIR /worker
COPY . .
RUN pip install --no-cache-dir redis celery requests
CMD ["celery", "--app=tasks", "worker", "--loglevel=info"]
