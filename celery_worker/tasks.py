
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
#from pydantic import BaseModel
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




def generate_summary(text, temperature, max_tokens, custom_prompt=None, chunk_summary=False, final_summary = False, whole_text = False, model_name = MODEL_NAME):
    """
    Generates summary based on arguments.
    """
    if custom_prompt:
        prompt = custom_prompt.replace("{text}", text)
    else:
        #TODO: UNCOMMENT WHEN META-PROMPT IS ENABLED
        #prompts = get_cached_prompts()
        # if isinstance(PROMPTS, str):
        #     print(PROMPTS)
        #     return "[SUMMARY_FAILED]", 0.0
        
        if chunk_summary:
            prompt = f"""You are an advanced IT developer team leader, an expert in recruiting IT professionals. Your goal is to write a structured summary of a part of a job interview given in a form of a meeting transcript in russian language, focusing only on the candidate's answers and narrative.\n\n## Principles for creating the summary:\n- Record only information from the candidate\n- Do not include job descriptions, company information, or conditions mentioned by the recruiter\n- Maintain the natural sequence of the conversation\n- Use russian language similar to the author's original style\n\n## Working process:\n1. Carefully study the given part of the interview transcript in russian\n2. Identify the names of the participants and their roles: recruter is asking questions, candidate is answering and telling about his experience\n3. Identify all topics discussed during the interview\n4. For each topic:\n   - Write its title\n   - Identify subtopics\n   - Present the content as close as possible to the candidate's original response\n   - Include specific examples and situations\n5. Check the completeness and accuracy of the information from the point of view of IT professional. \n\n## Summary structure:\n\n### Interview participants:\n- Names and roles of participants\n\n### Main content:\nDivide by topics, for example:\n- Work experience\n- Technical experience\n- Professional achievements\n- Reasons for job search\n- Personal and communication skills\n- etc.\n\nFor each topic:\n- Topic title\n- Subtopics\n- Detailed presentation of the candidate's answers\n- Examples from their experience\n\nBe careful not to mix people mentioned in the transcript with candidate.\n\nGive your answer in russian.

Part of the interview:\n\n{text}"""
        elif final_summary:
            model_name = "qwen2.5:32b"
            if whole_text:
                # TODO: change prompt for whole text for job interviews
                prompt = f"""Внимательно изучи и сделай резюме транскрипта записи встречи. Во-первых, выяви участников встречи. Не путай участников встречи между собой, используй логику встречи, чтобы точнее определить участников. Затем, определи основные тезисы, которые обсуждались во время встречи, запиши протокол встречи на основе представленного транскрипта по следующему формату:
\t1. 10 ключевых тезисов встречи
\t2. Принятые решения, ответственные за их исполнения, сроки
\t3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

Транскрипт встречи:\n\n{text}"""
            else:
                prompt = f"""#You are an experienced IT developers team leader, expert in recruitment of IT professionals in your team. Your goal is to produce a report about candidate's strengths and weaknesses.\n#Synthesize the following chunk summaries of a job interview given in russian into a single, cohesive analysis, ensuring no loss of critical details of the meeting. \nFirst, identify the participants' names, extract their roles. Focus on the candidate only. \nUse the logic of the meeting and the roles of participants to avoid mistakes.\n\n## Principles for creating the summary:\n\n- Record only information from the candidate\n- Do not include job descriptions, company information, or conditions mentioned by the recruiter\n- Maintain the natural sequence of the conversation\n- Use russian language similar to the author's original style\n\n## Working process:\n\n1. Carefully study the transcript summaries in russian\n2. Identify the names of the participants and their roles\n3. Identify all topics discussed during the interview\n4. For each topic:\n   - Write its title\n   - Identify subtopics\n   - Present the content as close as possible to the candidate's original text\n   - Include specific examples and situations important for candidate assessment  \n5. Check the completeness and accuracy of the information\n\n## Summary structure:\n\n### Interview participants:\n- Names and roles of participants\n\n### Main content:\nDivide by topics, for example:\n- Work experience\n- Technical experience\n- Professional achievements\n- Reasons for job search\n- Personal and communication skills\n- etc.\n\nFor each topic:\n- Topic title\n- Subtopics\n- Detailed presentation of the candidate's answers\n- Examples from their experience\n\n### Overall conclusion:\n<General conclusion about the candidate's competencies>\n\n### Strengths:\n<Candidate's strengths>\n\n### Weaknesses:\n<Candidate's weaknesses>\n\nGive your response in russian.

Chunk summaries:\n\n{text}"""

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

DATA_URL = "http://ai.rndl.ru:5017/api/data"
def send_results(test_result):
    try:
        response = requests.post(
            DATA_URL,
            headers={"Content-Type": "application/json"},
            data=test_result,
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

    whole_text_summary = params.get("checked", False)
    temp_final = params.get("temp_final", 0.6)
    max_tokens_final = params.get("max_tokens_final", 5000)
    final_prompt = params.get("final_prompt", None)

    if not whole_text_summary:
        # Chunk summary
        chunk_size = params.get("chunk_size", 1800)
        overlap = params.get("overlap", 0.3)
        temp_chunk = params.get("temp_chunk", 0.4)
        chunk_prompt = params.get("chunk_prompt", None)
        max_tokens_chunk = params.get("max_tokens_chunk", 1500)

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

        final_summary, final_time = generate_summary(combined_input, temp_final, max_tokens_final, final_prompt, final_summary=True) # final summary
    else:
        # No chunking summary
        final_summary, final_time = generate_summary(text, temp_final, max_tokens_final, final_prompt, final_summary=True, whole_text=True)

    #TODO: UNCOMMENT WHEN USING META-PROMPT  
    # Retrieve cached prompts for reporting
    # prompts = get_cached_prompts()
    # chunk_prompt_text = prompts.prompts[0]
    # final_prompt_text = prompts.prompts[1]
    final_msg = json.dumps({
        "version": 1.51,
        "description": "Саммари рекрутинговой встречи с обновленными промптами.",
        "type": "final",
        "Author": "ErnestSaak",
        "date_time": datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).strftime("%Y-%m-%d %H:%M:%S"),
        "document_url": "https://drive.google.com/file/d/1bRy761r67BlAwTZFP_gg-6xe6zmCSkSJ/view?usp=sharing",
        "chunk_model": MODEL_NAME,

        #CHANGE MODEL IF DIFFERENT FOR FINAL SUMMARY
        "final_model": "qwen2.:32b",
        "input_params": {
            "context_length": 32768,
            # TODO: UNCOMMENT WHEN USING META-PROMPT
            #"global_prompt": GLOBAL_PROMPT,
            #"meta_prompt": META_PROMPT,
            
            "chunk_prompt": None if whole_text_summary else """You are an advanced IT developer team leader, an expert in recruiting IT professionals. Your goal is to write a structured summary of a part of a job interview given in a form of a meeting transcript in russian language, focusing only on the candidate's answers and narrative.\n\n## Principles for creating the summary:\n- Record only information from the candidate\n- Do not include job descriptions, company information, or conditions mentioned by the recruiter\n- Maintain the natural sequence of the conversation\n- Use russian language similar to the author's original style\n\n## Working process:\n1. Carefully study the given part of the interview transcript in russian\n2. Identify the names of the participants and their roles: recruter is asking questions, candidate is answering and telling about his experience\n3. Identify all topics discussed during the interview\n4. For each topic:\n   - Write its title\n   - Identify subtopics\n   - Present the content as close as possible to the candidate's original response\n   - Include specific examples and situations\n5. Check the completeness and accuracy of the information from the point of view of IT professional. \n\n## Summary structure:\n\n### Interview participants:\n- Names and roles of participants\n\n### Main content:\nDivide by topics, for example:\n- Work experience\n- Technical experience\n- Professional achievements\n- Reasons for job search\n- Personal and communication skills\n- etc.\n\nFor each topic:\n- Topic title\n- Subtopics\n- Detailed presentation of the candidate's answers\n- Examples from their experience\n\nBe careful not to mix people mentioned in the transcript with candidate.\n\nGive your answer in russian.""",
            "final_summary_prompt": """#You are an experienced IT developers team leader, expert in recruitment of IT professionals in your team. Your goal is to produce a report about candidate's strengths and weaknesses.\n#Synthesize the following chunk summaries of a job interview given in russian into a single, cohesive analysis, ensuring no loss of critical details of the meeting. \nFirst, identify the participants' names, extract their roles. Focus on the candidate only. \nUse the logic of the meeting and the roles of participants to avoid mistakes.\n\n## Principles for creating the summary:\n\n- Record only information from the candidate\n- Do not include job descriptions, company information, or conditions mentioned by the recruiter\n- Maintain the natural sequence of the conversation\n- Use russian language similar to the author's original style\n\n## Working process:\n\n1. Carefully study the transcript summaries in russian\n2. Identify the names of the participants and their roles\n3. Identify all topics discussed during the interview\n4. For each topic:\n   - Write its title\n   - Identify subtopics\n   - Present the content as close as possible to the candidate's original text\n   - Include specific examples and situations important for candidate assessment  \n5. Check the completeness and accuracy of the information\n\n## Summary structure:\n\n### Interview participants:\n- Names and roles of participants\n\n### Main content:\nDivide by topics, for example:\n- Work experience\n- Technical experience\n- Professional achievements\n- Reasons for job search\n- Personal and communication skills\n- etc.\n\nFor each topic:\n- Topic title\n- Subtopics\n- Detailed presentation of the candidate's answers\n- Examples from their experience\n\n### Overall conclusion:\n<General conclusion about the candidate's competencies>\n\n### Strengths:\n<Candidate's strengths>\n\n### Weaknesses:\n<Candidate's weaknesses>\n\nGive your response in russian.""",
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
    }, indent=2)

    # Local save for the tests
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

    # SEND RESULTS TO DB
    send_results(final_msg)

    return final_msg

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
        # Chunking test with combinations
        combinations = json.loads(r.get(f"test:{task_id}:combinations"))
        if not combinations:
            logging.error(f"[ERROR] No combinations generated for task {task_id}.")
            return

        logging.info(f"\n\n[DEBUG] COMBINATIONS RECEIVED: {combinations}\n\n")

        test_count = 1
        for combination in combinations:
            logging.warning(f"\n[DEBUG] STARTING COMBINATION TEST #{test_count}\n")
            chunk_size, chunk_overlap, temp_chunk, temp_final = combination

            params_dict = {
                "chunk_size": chunk_size,
                "overlap": chunk_overlap / chunk_size,
                "temp_chunk": temp_chunk,
                "temp_final": temp_final,
                "max_tokens_chunk": 1500,
                "max_tokens_final": 5000
            }

            new_task_id = str(uuid.uuid4())
            r.set(f"summarize:{new_task_id}:text", text)
            r.set(f"summarize:{new_task_id}:params", json.dumps(params_dict))
            #celery.send_task("tasks.process_document", args=[new_task_id])
            try:
                process_document(new_task_id)
            except Exception as e:
                logging.exception(f"Failed to process combination #{test_count}: {e}")

            test_count += 1

        time.sleep(5)

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