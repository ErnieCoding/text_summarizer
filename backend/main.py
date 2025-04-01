
from flask import Flask, request, jsonify, stream_with_context, Response
from celery import Celery
import redis
import json
import uuid
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='redis', port=6379, decode_responses=True)

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
