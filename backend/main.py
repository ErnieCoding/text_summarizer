
from flask import Flask, request, jsonify, stream_with_context, Response
from celery import Celery
import redis
import json
import uuid
import os
from flask_cors import CORS
from itertools import product
import logging

app = Flask(__name__)
CORS(app)

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.route("/anonymize", methods=["GET"])
def anonymize():
    pass

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    if file.filename == "":
        return {"error": "Empty filename"}, 400

    params_json = request.form.get("params")
    if not params_json:
        return {"error": "Missing params"}, 400
    
    try:
        params = json.loads(params_json)
    except Exception as e:
        return {"error": f"Invalid params format: {str(e)}"}, 400
    
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    #Transcription task
    task_id = str(uuid.uuid4())
    r.set(f"transcribe:{task_id}:filename", file.filename)
    result = celery_app.send_task("tasks.transcribe_meeting", args=[task_id])
    response = result.get()

    if response["status"] == 200:
        return response["transcription"]
    else:
        return "ERROR TRANSCRIBING FILE"


@app.route("/test", methods=["POST"])
def test():
    data = request.get_json()
    text = data.get("text", "")

    # anonymize_task_id = str(uuid.uuid4())
    # r.set(f"test:{anonymize_task_id}:text", text)

    # try:
    #     result = celery_app.send_task("tasks.anonymize_text", args=[anonymize_task_id])
    #     result.get(timeout=30)

    #     text = r.get(f"test:{anonymize_task_id}:clean_text")
    #     reverse_patch = r.get(f"test:{anonymize_task_id}:reverse_patch")

    #     if not text or not reverse_patch:
    #         raise ValueError("No results found in Redis.")
        
    #     reverse_patch = json.loads(reverse_patch)
    # except Exception as e:
    #     return jsonify({"error": f"Anonymization task failed: {str(e)}"}), 500

    params = data.get("params", {})

    if not text or not params:
        return jsonify({"error": "No text or parameters provided"}), 400
    
    task_id = str(uuid.uuid4())
    r.set(f"test:{task_id}:text", text)

    if params.get("checked", False):
        # No chunking
        r.set(f"test:{task_id}:params", json.dumps(params))
    else:
        # Chunking
        try:
            combinations = list(product(
                [params["author"]],
                [params["chunkModel"]],
                [params["finalModel"]],
                params["chunk_size_range"],
                params["overlap"],
                params["temp_chunk"],
                params["temp_final"],
                [params["description"]],
                [params["chunk_prompt"]],
                [params["final_prompt"]]
            ))
            r.set(f"test:{task_id}:combinations", json.dumps(combinations))
        except KeyError as e:
            return jsonify({"error": f"Missing key for chunking: {e}"}), 400
    
    # r.set(f"test:{task_id}:reverse_patch", json.dumps(reverse_patch)) # reverse_patch for reversing anonymized text
    celery_app.send_task("tasks.test_params", args=[task_id])
    
    return jsonify({"task_id": task_id})


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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
