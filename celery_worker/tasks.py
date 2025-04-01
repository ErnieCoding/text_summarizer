
import os
import redis
import time
import json
import requests
from celery import Celery
import logging

# OLLAMA_URL = "http://ollama:11434/api/chat"
# MODEL_NAME = "llama3.1:8b"
OLLAMA_URL = "http://host.docker.internal:8003/v1/chat/completions"


CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='redis', port=6379, decode_responses=True)

def split_text(text, chunk_size=1800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def estimate_tokens(text):
    return int(len(text.split()) * 1.3)

def generate_summary(text, temperature, max_tokens, custom_prompt=None):
    if custom_prompt:
        prompt = custom_prompt.replace("{text}", text)
    else:
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    payload = {
        # "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        logging.info("Запуск задачки")
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        logging.info(response.json())
        result = response.json()
        elapsed = round(time.time() - start_time, 2)
        return result["choices"][0]['message']['content'].strip(), elapsed
    except Exception:
        return "[SUMMARY_FAILED]", 0.0

@celery.task(name="tasks.process_document")
def process_document(task_id):
    text = r.get(f"summarize:{task_id}:text")
    params = json.loads(r.get(f"summarize:{task_id}:params") or "{}")

    chunk_size = params.get("chunk_size", 1800)
    overlap = params.get("overlap", 200)
    temp_chunk = params.get("temp_chunk", 0.4)
    temp_final = params.get("temp_final", 0.6)
    max_tokens_chunk = params.get("max_tokens_chunk", 800)
    max_tokens_final = params.get("max_tokens_final", 2400)
    chunk_prompt = params.get("chunk_prompt", None)
    final_prompt = params.get("final_prompt", None)

    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    progress = []

    for i, chunk in enumerate(chunks):
        summary, duration = generate_summary(chunk, temp_chunk, max_tokens_chunk, chunk_prompt)
        progress.append({"chunk": i + 1, "summary": summary, "duration": duration})
        msg = json.dumps({
            "type": "chunk",
            "chunk": i + 1,
            "total": len(chunks),
            "summary": summary,
            "duration": duration
        })
        r.publish(f"summarize:{task_id}:events", msg)

    valid_chunks = [p for p in progress if p["summary"] and p["summary"] != "[SUMMARY_FAILED]"]
    combined_input = "\n\n".join([
        f"Chunk {i} Summary:\n{p['summary']}" for i, p in enumerate(valid_chunks, 1)
    ]) or "The document contains multiple summaries that need to be unified."

    final_summary, final_time = generate_summary(combined_input, temp_final, max_tokens_final, final_prompt)

    final_msg = json.dumps({
        "type": "final",
        "summary": final_summary,
        "duration": final_time,
        "token_count": estimate_tokens(text)
    })
    r.publish(f"summarize:{task_id}:events", final_msg)
