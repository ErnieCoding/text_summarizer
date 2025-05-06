# import whisper
# import os
# from celery import Celery
# import redis
# import logging

# CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
# CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
# r = redis.Redis(host='redis', port=6379, decode_responses=True)

# @celery.task(name="transcription.transcribe_file")
# def transcribe_file(task_id):
#     filename = r.get(f"transcribe:{task_id}:filename")
    
#     model = whisper.load_model("large")

#     try:
#         logging.info(f"Starting transcription of file: {filename}")
#         result = model.transcribe(filename)
#         return result["text"]
#     except Exception as e:
#         logging.info(f"Error transcribing file: {e}")
#         return ""

import whisper

def transcribe_file(filename):
    model = whisper.load_model("large")

    try:
        print("starting transcription")
        result = model.transcribe(filename)
        return result["text"]
    except Exception as e:
        print(f"\nError processing file {e}\n")
        return ""