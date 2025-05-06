FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask redis celery gunicorn flask-cors openai-whisper

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "1200", "main:app"]
