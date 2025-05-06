
from flask import Flask, request, jsonify, stream_with_context, Response
from celery import Celery
import redis
import json
import uuid
import os
from flask_cors import CORS
from itertools import product
import whisper
import torch
import logging

app = Flask(__name__)
CORS(app)

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='redis', port=6379, decode_responses=True)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    if file.filename == "":
        return {"error": "Empty filename"}, 400

    # params_json = request.form.get("params")
    # if not params_json:
    #     return {"error": "Missing params"}, 400
    
    # try:
    #     params = json.loads(params_json)
    # except Exception as e:
    #     return {"error": f"Invalid params format: {str(e)}"}, 400
    
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    logging.info(f"Saved uploaded file to: {file_path}")
    
    #Transcription
    try:
        logging.info("STARTING TRANSCRIPTION")

        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("large").to(device)

        result = model.transcribe(file_path, fp16=(device == "cuda"))

        logging.info("TRANSCRIPTION COMPLETED")
        return {"transcription": result["text"], "status": 200}
    except Exception as e:
        logging.exception(f"\nERROR PROCESSING FILE {file_path}: {e}\n")
        return {"transcription":"error", "status": 400} 


@app.route("/test", methods=["POST"])
def test():
    data = request.get_json()
    text = data.get("text", "")
    params = data.get("params", {})

    if not text or not params:
        return jsonify({"error": "No text or parameters provided"}), 400
    
    combinations = list(product(
    params["chunk_size_range"],
    params["overlap"],
    params['temp_chunk'],
    params["temp_final"],
    ))

    task_id = str(uuid.uuid4())
    r.set(f"test:{task_id}:text", text)
    r.set(f"test:{task_id}:combinations", json.dumps(combinations))
    celery_app.send_task("tasks.test_params", args=[task_id])

    return jsonify({"task_id":task_id})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    params = data.get("params", {})
    if not text:
        return jsonify({"error": "No text provided"}), 400

    task_id = str(uuid.uuid4())
    r.set(f"summarize:{task_id}:text", text)
    r.set(f"summarize:{task_id}:params", json.dumps(params))
    celery_app.send_task("tasks.process_document", args=[task_id])
    return jsonify({"task_id": task_id})

@app.route("/stream/<task_id>", methods=["GET"])
def stream_summary(task_id):
    def event_stream():
        pubsub = r.pubsub()
        pubsub.subscribe(f"summarize:{task_id}:events")
        for message in pubsub.listen():
            if message['type'] == 'message':
                yield f"data: {message['data']}\n\n"
                if "\"type\": \"final\"" in message['data']:
                    break
        pubsub.close()
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')
