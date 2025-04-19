FROM python:3.10-slim


WORKDIR /tasks


COPY . .


RUN pip install --no-cache-dir \
    celery \
    redis \
    requests \
    tiktoken \
    pypdf \
    pydantic

CMD ["celery", "--app=tasks", "worker", "--loglevel=info"]
