FROM python:3.10-slim


WORKDIR /tasks


COPY . .

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    celery \
    redis \
    requests \
    tiktoken \
    pypdf \
    pydantic \
    openai-whisper

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

CMD ["celery", "--app=tasks", "worker", "--loglevel=info"]
