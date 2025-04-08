
import os
import redis
import time
import json
import requests
from celery import Celery
import logging
import re
from tokenCounter import count_tokens
import datetime
import zoneinfo

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = "phi4:14b"
#OLLAMA_URL = "http://host.docker.internal:8003/v1/chat/completions"


CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='redis', port=6379, decode_responses=True)

def split_text(text, chunk_size=1800, overlap=0.3):
    """
    Splits provided text by the number of tokens with overlapping regions.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    curr_chunk = []
    curr_length = 0

    for sentence in sentences:
        sentence_length = count_tokens(text=sentence)

        if sentence_length + curr_length <= chunk_size:
            curr_chunk.append(sentence)
            curr_length += sentence_length
        else:
            chunks.append(" ".join(curr_chunk))

            overlap_size = int(overlap * chunk_size)

            retained_tokens = []
            retained_length = 0
            while curr_chunk and retained_length < overlap_size:
                retained_sentence = curr_chunk.pop()
                retained_tokens.insert(0, retained_sentence)
                retained_length += count_tokens(text=retained_sentence)

            curr_chunk = retained_tokens + [sentence]
            curr_length = retained_length + sentence_length

    if curr_chunk:
        chunks.append(" ".join(curr_chunk))

    return chunks

def generate_summary(text, temperature, max_tokens, custom_prompt=None, chunk_summary=False, final_summary = False):
    if custom_prompt:
        prompt = custom_prompt.replace("{text}", text)
    elif chunk_summary:
        prompt = f"""Summarize this text chunk clearly and accurately. Include:

1. Main plot developments — What happens in this section?
2. Character progression and relationships — How do key characters act, change, reveal themselves, and interact with one another?
3. Avoid unnecessary detail or repetition. Focus on what matters for understanding the story.

{text}
"""
    else:
        prompt = f"""Synthesize the following chunk summaries into a single, cohesive analysis of the text while ensuring no loss of critical details of the plot, characters, etc. Do not provide any other information in your answer except described above holistic summary of the whole text.
- Eliminate redundant information and merge similar themes.
- Identify overarching patterns and insights that emerge when considering the full text holistically.

Provide a final summary that includes:
1. The complete plot progression from start to finish, capturing all key events and details
2. All character progression, interactions and relationships.

{text}
"""
    model_name = MODEL_NAME
    if final_summary:
        model_name = "llama3.1:8b"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False
    }

    try:
        logging.info("Starting summary generation via API")
        start_time = time.time()
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()

        content = response.json()["response"].strip()

        elapsed = round(time.time() - start_time, 2)
        return content, elapsed
    except Exception as e:
        logging.exception(f"Summary generation failed: {e}")
        return "[SUMMARY_FAILED]", 0.0
    
def get_context_length():
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": MODEL_NAME}
        )
        response.raise_for_status()
        model_info = response.json()

        if "model_info" in model_info:
            model_info_fields = model_info["model_info"]
            if "llama.context_length" in model_info_fields:
                return model_info_fields["llama.context_length"]
            
        return None
    except Exception as e:
        print(f"Failed to fetch model context length: {e}")
        return None


@celery.task(name="tasks.process_document")
def process_document(task_id):
    text = r.get(f"summarize:{task_id}:text")
    params = json.loads(r.get(f"summarize:{task_id}:params") or "{}")

    chunk_size = params.get("chunk_size", 1800)
    overlap = params.get("overlap", 0.3)
    temp_chunk = params.get("temp_chunk", 0.4)
    temp_final = params.get("temp_final", 0.6)
    max_tokens_chunk = params.get("max_tokens_chunk", 1500)
    max_tokens_final = params.get("max_tokens_final", 5000)
    chunk_prompt = params.get("chunk_prompt", None)
    final_prompt = params.get("final_prompt", None)

    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    progress = []

    sum_token_responses = 0
    chunk_summary_duration = 0
    for i, chunk in enumerate(chunks):
        summary, duration = generate_summary(chunk, temp_chunk, max_tokens_chunk, chunk_prompt, chunk_summary=True)
        
        chunk_summary_duration += duration
        sum_token_responses += count_tokens(text=summary)

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

    final_summary, final_time = generate_summary(combined_input, temp_final, max_tokens_final, final_prompt, final_summary=True)

    final_msg = json.dumps({
        "type": "final",
        "Author": "ErnestSaak",
        "date_time": datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).strftime("%Y-%m-%d %H:%M:%S"),
        "document_url": "https://drive.google.com/file/d/1pbcOsUMlzJD81rEjJ-g5_Uu1DPdlnADw/view?usp=sharing",
        "chunk_model": MODEL_NAME,
        "final_model": "llama3.1:8b",
        "input_params": {
            "context_length": get_context_length(),
            "chunk_prompt": """Summarize this text chunk clearly and accurately. Include:

1. Main plot developments — What happens in this section?
2. Character progression and relationships — How do key characters act, change, reveal themselves, and interact with one another?
3. Avoid unnecessary detail or repetition. Focus on what matters for understanding the story.""",
            "final_summary_prompt": """Synthesize the following chunk summaries into a single, cohesive analysis of the text while ensuring no loss of critical details of the plot, characters, etc. Do not provide any other information in your answer except described above holistic summary of the whole text.
- Eliminate redundant information and merge similar themes.
- Identify overarching patterns and insights that emerge when considering the full text holistically.

Provide a final summary that includes:
1. The complete plot progression from start to finish, capturing all key events and details
2. All character progression, interactions and relationships.""",
            "temp_chunk": temp_chunk,
            "temp_final": temp_final,
            "chunk_size": chunk_size,
            "chunk_overlap(tokens)": overlap * chunk_size, 
            "chunk_output_limit(tokens)": max_tokens_chunk,
            "final_output_limit(tokens)": max_tokens_final,
        },
        "output_params":{
            "num_chunks": len(chunks),
            "avg_chunk_output(tokens)": sum_token_responses // len(chunks),
            "avg_chunk_summary_time(sec)": round(chunk_summary_duration / len(chunks), 2), 
            "final_response(tokens)": count_tokens(text=final_summary),
        },
        "summary": final_summary,
        "total_time(sec)": round(final_time + chunk_summary_duration, 2),
        "text_token_count": count_tokens(text=text)
    }, indent=2)

    TESTS_DIR = "/tasks/tests"
    os.makedirs(TESTS_DIR, exist_ok=True)

    filename = f"final_{task_id}.json"
    file_path = os.path.join(TESTS_DIR, filename)

    logging.warning(f"[DEBUG] __file__ = {__file__}")
    logging.warning(f"[DEBUG] TESTS_DIR = {TESTS_DIR}")
    logging.warning(f"[DEBUG] Writing to file path: {file_path}")

    try:
        with open(file_path, "w+") as file:
            file.write(final_msg)
        logging.info(f"[WRITE] Final summary saved to {file_path}")
    except Exception as e:
        logging.exception(f"Failed to write final summary to file: {e}")

    r.publish(f"summarize:{task_id}:events", final_msg)

    return final_msg
