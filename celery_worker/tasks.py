
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
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = "qwen2.5:14b"
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

#TODO: refactor meta-prompt to include company dictionary
#TODO: implement a better generation strategy that doesn't limit generation creativity for prompts
def get_prompts(model_name = MODEL_NAME):
    """
    Generates prompt for chunk and final summaries using Ollama structured outputs in JSON format
    """

    # Глобальный промпт
    global_prompt = """Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи на основе представленного транскрипта по следующему формату:
1. 10 ключевых тезисов встречи
2. Принятые решения, ответственные за их исполнения, сроки
3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач."""

    # Мета-промпт
    meta_prompt = f"""Ниже приведён глобальный промпт, который хорошо работает для анализа полной стенограммы встречи. Преобразуй его в два отдельных промпта без дополнительных инструкций, шагов или полей:
1.Промпт для обработки одного чанка текста (фрагмента транскрипта):
\t-Он должен ограничиваться только предоставленным фрагментом, не делать глобальных выводов, извлекать локальные тезисы, задачи, решения, имена участников.

2.Промпт для финальной агрегации:
\t-Он должен принимать уже обработанные чанки (в виде кратких тезисов, задач и решений) и собирать на их основе финальный протокол встречи — с обобщением, исключением повторов, уточнением сроков и распределением задач. Цель — сохранить смысл и структуру оригинального промпта, но адаптировать его к двухэтапной логике обработки транскрипта.

Верни результат **строго в следующем формате**:

{{
  "prompts": [
    "Промпт для обработки одного чанка текста...",
    "Промпт для финальной агрегации..."
  ]
}}

Не добавляй никаких других полей, описаний, нумерации или комментариев. Только JSON-объект с ключом "prompts".

Вот глобальный промпт:
{global_prompt}"""

    class Prompts(BaseModel):
        prompts: list[str]
    
    payload = {
        "model": model_name,
        "prompt": meta_prompt,
        "temperature": 0.6,
        "stream": False,
        "format": "json"
    }

    try:
        ollama = "http://ollama:11434"
        response = requests.post(f"{ollama}/api/generate", json=payload)
        response.raise_for_status()

        raw_response = response.json()["response"]
        print(f"Raw response:\n{raw_response}\n")

        prompts = Prompts.model_validate_json(raw_response)

        return prompts
    except Exception as e:
        logging.info(f"\nError generating prompts: {e}\n")
        print(e)
        return "[PROMPT GENERATION FAILED]"
    
PROMPTS = None
def get_cached_prompts():
    """
    Generates and caches prompts to retrieve during generation
    """
    global PROMPTS
    if PROMPTS is None or isinstance(PROMPTS, str):
        PROMPTS = get_prompts()
    return PROMPTS

def generate_summary(text, temperature, max_tokens, custom_prompt=None, chunk_summary=False, final_summary = False, model_name = MODEL_NAME):
    if custom_prompt:
        prompt = custom_prompt.replace("{text}", text)
    else:
        prompts = get_cached_prompts()
        if isinstance(PROMPTS, str):
            print(PROMPTS)
            return "[SUMMARY_FAILED]", 0.0
        
        if chunk_summary:
            prompt = f"""{prompts.prompts[0]}\n\nФрагмент:\n{text}"""
        elif final_summary:
            prompt = f"""{prompts.prompts[1]}\n\n{text}"""


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
        "version": 1.0,
        "description": "Имплементация мета-промптов для whisper транскриптов",
        "type": "final",
        "Author": "ErnestSaak",
        "date_time": datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).strftime("%Y-%m-%d %H:%M:%S"),
        "document_url": "https://drive.google.com/file/d/1bRy761r67BlAwTZFP_gg-6xe6zmCSkSJ/view?usp=sharing",
        "chunk_model": MODEL_NAME,

        #CHANGE MODEL IF DIFFERENT FOR FINAL SUMMARY
        "final_model": MODEL_NAME,
        "input_params": {
            "context_length": 32768,

            #TODO: dynamically retrieve generated prompts
            "chunk_prompt": """СДЕЛАЮ ПОЗЖЕ""",
            "final_summary_prompt": """СДЕЛАЮ ПОЗЖЕ""",
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
