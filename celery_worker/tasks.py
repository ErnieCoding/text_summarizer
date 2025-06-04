
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
import uuid
import whisper, torch

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

# Variables to cache selected prompts
CHUNK_PROMPT = ""
FINAL_PROMPT = ""
def generate_summary(text, temperature, max_tokens, finalModel = None, chunkModel = None, custom_prompt=None, chunk_summary=False, final_summary = False, whole_text = False):
    """
    Generates summary based on parameters.
    """
    global CHUNK_PROMPT, FINAL_PROMPT

    if custom_prompt and custom_prompt.strip():
        if chunk_summary:
            CHUNK_PROMPT = custom_prompt
            model_name = chunkModel
        elif final_summary:
            FINAL_PROMPT = custom_prompt
            model_name = finalModel

        prompt = custom_prompt.replace("{text}", text)
    else: # NOT UPDATED
        if chunk_summary:
            model_name = chunkModel
            prompt = f"Summarize the following transcript:\n\n{text}"
            CHUNK_PROMPT = prompt
        elif final_summary:
            model_name = finalModel
            if whole_text:
                prompt = f"Summarize the following transcript:\n\n{text}"
            else:
                prompt = f"Summarize the following transcript:\n\n{text}"
            FINAL_PROMPT = prompt

    if chunk_summary:
        num_ctx = count_tokens(prompt) + 1000
    elif final_summary:
        num_ctx = 40000

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2500,
            "num_ctx": num_ctx,
        }
    }

    try:
        logging.info(f"Starting summary generation with num_ctx: {num_ctx}")
        start_time = time.time()
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()

        content = response_json.get("response").strip()

        elapsed = round(time.time() - start_time, 2)

        prompt_eval_count = response_json.get("prompt_eval_count")
        logging.info(f"TOKENS PROCESSED: {prompt_eval_count}")
        return content, elapsed
    except Exception as e:
        logging.exception(f"Summary generation failed: {e}")
        return "[SUMMARY_FAILED]", 0.0

DATA_URL = "http://llm.rndl.ru:5017/api/data"
def send_results(test_result):
    try:
        safe_json = json.dumps(test_result, ensure_ascii=False)
        response = requests.post(
            DATA_URL,
            data=safe_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        logging.info(f"[UPLOAD RESPONSE] {response.status_code} - {response.text}")
        response.raise_for_status()
        logging.info(f"[UPLOAD] Successfully sent summary to {DATA_URL}. Response: {response.text}")
    except Exception as e:
        logging.exception(f"[UPLOAD ERROR] {e}")

@celery.task(name="tasks.process_document")
def process_document(task_id):
    text = r.get(f"summarize:{task_id}:text")
    params = json.loads(r.get(f"summarize:{task_id}:params") or "{}")

    # Initial params
    whole_text_summary = params.get("checked", False)
    temp_final = params.get("temp_final", 0.6)
    max_tokens_final = params.get("max_tokens_final", 5000)
    final_prompt = params.get("final_prompt", None)
    test_author = params.get("author", "RConf")
    test_description = params.get("description", "")
    finalModel = params.get("finalModel", "qwen2.5:32b")

    if not whole_text_summary:
        # Chunk params
        chunk_size = params.get("chunk_size", 1800)
        overlap = params.get("overlap", 0.3)
        temp_chunk = params.get("temp_chunk", 0.4)
        chunk_prompt = params.get("chunk_prompt", None)
        max_tokens_chunk = params.get("max_tokens_chunk", 1500)
        chunkModel = params.get("chunkModel", "qwen2.5:14b")

        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        progress = []
        
        sum_token_responses = 0
        chunk_summary_duration = 0
        for i, chunk in enumerate(chunks):
            summary, duration = generate_summary(
                text=chunk, 
                temperature=temp_chunk, 
                max_tokens=max_tokens_chunk, 
                custom_prompt=chunk_prompt, 
                chunk_summary=True, 
                chunkModel=chunkModel
            )
            
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

        final_summary, final_time = generate_summary( # final summary
            text=combined_input, 
            finalModel=finalModel, 
            temperature=temp_final, 
            max_tokens=max_tokens_final, 
            custom_prompt=final_prompt, 
            final_summary=True
        )
    else:
        # No chunking summary
        final_summary, final_time = generate_summary(
            text=text, 
            finalModel=finalModel, 
            temperature=temp_final, 
            max_tokens=max_tokens_final, 
            custom_prompt=final_prompt, 
            final_summary=True, 
            whole_text=True
        )

    final_data = {
        "version": 2.3,
        "description": test_description,
        "type": "final",
        "Author": test_author,
        "date_time": datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_model": None if whole_text_summary else chunkModel,
        "final_model": finalModel,
        "input_params": {
            "context_length": 32768,
            "chunk_prompt": None if whole_text_summary else CHUNK_PROMPT,
            "final_summary_prompt": FINAL_PROMPT,
            "temp_chunk": None if whole_text_summary else temp_chunk,
            "temp_final": temp_final,
            "chunk_size": None if whole_text_summary else chunk_size,
            "chunk_overlap(tokens)": None if whole_text_summary else overlap * chunk_size, 
            "chunk_output_limit(tokens)": None if whole_text_summary else max_tokens_chunk,
            "final_output_limit(tokens)": max_tokens_final,
        },
        "output_params":{
            "num_chunks": None if whole_text_summary else len(chunks),
            "avg_chunk_output(tokens)": None if whole_text_summary else sum_token_responses // len(chunks),
            "avg_chunk_summary_time(sec)": None if whole_text_summary else round(chunk_summary_duration / len(chunks), 2), 
            "final_response(tokens)": count_tokens(text=final_summary),
        },
        "summary": final_summary,
        "total_time(sec)": round(final_time, 2) if whole_text_summary else round(final_time + chunk_summary_duration, 2),
        "text_size": count_tokens(text=text)
    }
    final_msg = json.dumps(final_data, ensure_ascii=False, indent=2)

    # Local save for the tests
    if final_data["summary"] == "[SUMMARY_FAILED]":
        logging.error("SUMMARY FAILED, ABORTING SAVE")
        return ""
    else:
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

        # CURRENTLY NOT WORKING
        # SEND RESULTS TO DB
        # send_results(final_data)

    return final_msg

# Batch runner
def run_test_batch(task_id):
    text = r.get(f"test:{task_id}:text")
    combinations = json.loads(r.get(f"test:{task_id}:combinations"))

    test_count = 1
    for combo in combinations:
        author, chunkModel, finalModel, chunk_size, chunk_overlap, temp_chunk, temp_final, description, chunk_prompt, final_prompt = combo

        params = {
            "author": author,
            "chunkModel": chunkModel,
            "finalModel": finalModel,
            "chunk_size": chunk_size,
            "overlap": chunk_overlap / chunk_size,
            "temp_chunk": temp_chunk,
            "temp_final": temp_final,
            "max_tokens_chunk": 1500,
            "max_tokens_final": 5000,
            "description": description,
            "chunk_prompt": chunk_prompt,
            "final_prompt": final_prompt
        }

        new_task_id = str(uuid.uuid4())
        r.set(f"summarize:{new_task_id}:text", text)
        r.set(f"summarize:{new_task_id}:params", json.dumps(params))

        try:
            logging.info(f"[BATCH] Waiting for test #{test_count} to finish")
            process_document(new_task_id)
        except Exception as e:
            logging.exception(f"[BATCH ERROR] Combination #{test_count} failed: {e}")

        test_count += 1

@celery.task(name="tasks.test_params")
def test_params(task_id):
    text = r.get(f"test:{task_id}:text")

    if r.exists(f"test:{task_id}:params"):
        # No chunking test
        params = json.loads(r.get(f"test:{task_id}:params"))
        if not params:
            logging.error(f"[ERROR] Empty params for task {task_id}")
            return

        logging.info(f"\n\n[DEBUG] PARAMS RECEIVED (NO CHUNKING): {params}\n\n")

        new_task_id = str(uuid.uuid4())
        r.set(f"summarize:{new_task_id}:text", text)
        r.set(f"summarize:{new_task_id}:params", json.dumps(params))
        celery.send_task("tasks.process_document", args=[new_task_id])

    elif r.exists(f"test:{task_id}:combinations"):
        logging.info(f"[DEBUG] Using run_test_batch for combinations")
        run_test_batch(task_id)
    else:
        logging.error(f"[ERROR] Neither params nor combinations found for task {task_id}")


@celery.task(name="tasks.transcribe_meeting")
def transcribe_meeting(task_id):
    filename = r.get(f"transcribe:{task_id}:filename")
    filepath = os.path.join("/shared/uploads", filename)

    try:
        
        logging.info(f"Saved uploaded file to: {filepath}")
        logging.info("STARTING TRANSCRIPTION")

        logging.info(f"\nPyTorch version: {torch.__version__}\n")
        logging.info(f"\nCUDA version: {torch.version.cuda}\n")
        logging.info(f"\nCUDA available: {torch.cuda.is_available()}\n")
        logging.info(f"\ncuDNN enabled: {torch.backends.cudnn.enabled}\n")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base").to(device)

        result = model.transcribe(filepath, fp16=(device == "cuda"))

        output_path = os.path.join("/shared/transcripts", f"{filename}.txt")
        with open(output_path, "w+", encoding="utf-8") as filewrite:
            filewrite.write(result["text"])

        logging.info(f"TRANSCRIPTION COMPLETED FOR FILE: {filepath}")
        return {"transcription": result["text"], "status": 200}
    except Exception as e:
        logging.exception(f"\nERROR PROCESSING FILE {filepath}: {e}\n")
        return {"transcription":"error", "status": 400}
    




#from pydantic import BaseModel
#TODO: UNCOMMENT FOR META-PROMPT IMPLEMENTATION

# GLOBAL_PROMPT = """Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи на основе представленного транскрипта по следующему формату:
# 1. 10 ключевых тезисов встречи
# 2. Принятые решения, ответственные за их исполнения, сроки
# 3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач."""

# META_PROMPT = f"""Ты- опытный менеджер по управлению разработкой, специалист по формированию и документированию заданий для разработчиков. Твоя цель - создать промпты для формирования максимально детально описанных заданий на разработку каждому участнику встречи разработчиков, не пропустив ни одной задачи и ни одного участника. В промптах нужно обращать внимание на обсуждаемые технические проблемы (например, "на Маке у клиентов есть проблема с микрофоном") и давать инструкции по указанию этих проблем в отчете, если это релевантно, соответсвующие задачи по исправлению выявленных проблем, если это упоминается в транскрипте. Ниже приведён глобальный промпт, который хорошо работает для анализа полной стенограммы встречи. Преобразуй его в два отдельных промпта без дополнительных инструкций, шагов или полей:
# 1.Промпт для обработки одного чанка текста (фрагмента транскрипта):
# \t-Он должен ограничиваться только предоставленным фрагментом, не делать глобальных выводов, извлекать локальные тезисы, задачи, решения, имена участников.

# 2.Промпт для финальной агрегации:
# \t-Он должен принимать уже обработанные чанки (в виде кратких тезисов, задач и решений) и собирать на их основе финальный протокол встречи — с обобщением, исключением повторов, уточнением сроков и распределением задач. Цель — сохранить смысл и структуру оригинального промпта, но адаптировать его к двухэтапной логике обработки транскрипта. Финальный промпт должен иметь содержать все пункты из глобального промпта

# Верни результат **строго в следующем формате**:

# {{
#   "prompts": [
#     "Промпт для обработки одного чанка текста...",
#     "Промпт для финальной агрегации..."
#   ]
# }}

# Не добавляй никаких других полей, описаний, нумерации или комментариев. Только JSON-объект с ключом "prompts".

# Вот глобальный промпт:
# {GLOBAL_PROMPT}"""

#TODO: implement a better generation strategy that doesn't limit generation creativity for prompts
# def get_prompts(model_name = MODEL_NAME):
#     """
#     Generates prompt for chunk and final summaries using Ollama structured outputs in JSON format
#     """

#     # Мета-промпт
#     meta_prompt = META_PROMPT

#     class Prompts(BaseModel):
#         prompts: list[str]
    
#     payload = {
#         "model": model_name,
#         "prompt": meta_prompt,
#         "temperature": 0.6,
#         "stream": False,
#         "format": "json"
#     }

#     try:
#         ollama = "http://ollama:11434"
#         response = requests.post(f"{ollama}/api/generate", json=payload)
#         response.raise_for_status()

#         raw_response = response.json()["response"]
#         print(f"\nRaw response:\n{raw_response}\n")

#         prompts = Prompts.model_validate_json(raw_response)

#         return prompts
#     except Exception as e:
#         logging.info(f"\nError generating prompts: {e}\n")
#         print(e)
#         return "[PROMPT GENERATION FAILED]"
    
# PROMPTS = None
# def get_cached_prompts():
#     """
#     Generates and caches prompts to retrieve during generation
#     """
#     global PROMPTS
#     if PROMPTS is None or isinstance(PROMPTS, str):
#         PROMPTS = get_prompts()
#     return PROMPTS