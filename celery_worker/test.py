import redis
from celery import Celery
import time
import os
from itertools import product
import uuid
import json
from rouge_score import rouge_scorer
import logging
import requests
import random
from tokenCounter import count_tokens
#from transcription import transcribe_file

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSCRIPT_TEXT_FILE = os.path.join(BASE_DIR, "transcribed_dev_meeting.txt")

with open(TRANSCRIPT_TEXT_FILE, "r", encoding="utf-8") as file:
    TRANSCRIPT_TEXT = file.read()

#AUDIO_FILE = os.path.join(BASE_DIR, "Dev_Meeting_Audio.mp3")

DATA_URL = "http://ai.rndl.ru:5017/api/data"

# MULTIPLE TESTS PARAMS
# params = {
#     "chunk_size":[5000, 5500, 6000, 6500, 7000], 
#     "chunk_overlap": [500], 
#     "temp_chunk": [0.2, 0.3, 0.4], 
#     "temp_final": [0.4, 0.5, 0.6]
#     }

params = {
    "chunk_size":[count_tokens(text=TRANSCRIPT_TEXT)], 
    "chunk_overlap": [1000], 
    "temp_chunk": [0.2, 0.3, 0.4], 
    "temp_final": [0.4, 0.5, 0.6]
    }


# SINGLE TEST PARAMS
# params = {
#     "chunk_size":[5000], 
#     "chunk_overlap": [500], 
#     "temp_chunk": [0.2], 
#     "temp_final": [0.3]
#     }

combinations = list(product(
    params["chunk_size"],
    params['chunk_overlap'],
    params['temp_chunk'],
    params["temp_final"]
))

#combinations = random.sample(combinations, 20) # complete random 20 tests

# def run_eval(data):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
#     reference = """# Meeting minutes

# ## 1. Key points of the meeting:
# 1. On Alfa-Bank: the status is positive, the information security specialists have approved the proposal, approval by the management is expected on December 24-26. The expected contract amount is 1.6-2 million rubles.
# 2. On OMS: a commercial proposal has been prepared for 2,500 users in messenger and videoconferencing mode with a limitation on report generation for 4 million rubles, plus a separate track for mass recruitment (about 500 thousand rubles).
# 3. Technodrone has expanded licenses from 10 to 30, having paid 120 thousand rubles.
# 4. On Astra: a pilot is starting with the Head of Recruitment and HRD, who will test the system in top positions. Prompts have been prepared to assess the candidates' compliance with the company's cultural values.
# 5. On MTS: the pilot has ended, a meeting has been scheduled to discuss the results with the heads of recruitment. Potential volume - 263 recruiters.
# 6. On Dobrotsen: a pilot assessment of four regional directors is planned for December 19-20. Prompts have been prepared taking into account their documentation and cases.
# 7. On Norilsk Nickel: a demo module has been prepared consisting of three autotests, two interviews to assess skills and a voice assistant.
# 8. Technical improvements for the voice assistant were discussed - simplifying authorization and saving user data.
# 9. The problem with group rooms on Russian servers has been solved by using the default presentation.
# 10. On the situation with MTUCI: an activation letter and a negotiation strategy with three possible scenarios have been prepared.

# ## 2. Decisions made, responsible persons, deadlines:
# 1. **Artem**:
# - Wait for the results of the approval with Alfa-Bank by December 26
# - Meet with OMS on the CP (deadline: tomorrow)
# - Agree the TOR for mass recruitment with OMS (deadline: today at [14:00]
# )
# - Conduct training for Astra (deadline: today at [12:00]
# )
# - Prepare 30 questions for profile processing (deadline: today)

# 2. **Sasha**:
# - Conduct a meeting with MTS on the pilot results (deadline: tomorrow at [12:30]
# )
# - Prepare a specific CP for MTS taking into account their needs
# - Conduct training for Dobrotsen (deadline: today at [12:00]
# )
# - Accompany the assessment at Dobrotsen on December 19-20
# - Clarify with Dobrotsen the time that they are putting it on reporting

# 3. **Max**:
# - Refine the voice assistant for Norilsk Nickel - implement saving user data in the browser (deadline: today)
# - Continue working on stabilizing the server infrastructure

# 4. **Lena**:
# - Send an activation letter to MTUCI (deadline: today after the meeting)
# - Prepare and agree on an informal letter for MTUCI with a proposal for solutions to the situation

# ## 3. Next steps:
# 1. **Urgent (today):**
# - Artem conducts training for Astra at [12:00]

# - Sasha conducts training for Dobrotsen at [12:00]

# - Artem coordinates the terms of reference for mass recruitment with OMS at [14:00]

# - Lena sends an activation letter to MTUCI
# - Max implements improvements to the voice assistant

# 2. **Tomorrow:**
# - Artem meets with OMS on CP
# - Sasha holds a meeting with MTS on the results of the pilot at [12:30]

# - Preparation for the assessment in Dobrotsen on December 19-20

# 3. **By the end of the week:**
# - Prepare a specific CP for MTS taking into account their needs and scale (263 recruiters)
# - Complete preparation for the assessment in Dobrotsen
# - Monitor the situation with MTUCI after sending letters"""
#     scores = scorer.score(reference, data['summary'])
#     rouge_l_f1 = scores['rougeL'].fmeasure
    
#     return rouge_l_f1

@celery.task(name="tasks.test_params")
def test_params(combinations):
    text = r.get(f"test:{task_id}:text")
    combinations = json.loads(r.get(f"test:{task_id}:combinations"))
    #text = TRANSCRIPT_TEXT

    # text = transcribe_file("Dev_Meeting_Audio.mp3")

    # with open("transcribed_dev_meeting.txt", "w+", encoding="utf-8") as file: 
    #     file.write(text)

    for combination in combinations:
        chunk_size, chunk_overlap, temp_chunk, temp_final = combination

        task_id = str(uuid.uuid4())

        params_dict = {
            "chunk_size": chunk_size,
            "overlap": chunk_overlap / chunk_size,
            "temp_chunk": temp_chunk,
            "temp_final": temp_final,
            "max_tokens_chunk": 1500,
            "max_tokens_final": 5000
        }

        r.set(f"summarize:{task_id}:text", text)       
        r.set(f"summarize:{task_id}:params", json.dumps(params_dict))

        result = celery.send_task("tasks.process_document", args=[task_id])

        summary_output = result.get()
        summary_dict = json.loads(summary_output)
        #f1_score = run_eval(summary_dict)
        summary_dict["f1_score"] = "undefined"

        updated_final_output = json.dumps(summary_dict, indent=2)

        try:
            response = requests.post(
                DATA_URL,
                headers={"Content-Type": "application/json"},
                data=updated_final_output,
            )
            response.raise_for_status()
            logging.info(f"[UPLOAD] Successfully sent summary to {DATA_URL}. Response: {response.text}")
        except Exception as e:
            logging.info(f"Error uploading data: {e}")

        time.sleep(5)

if __name__ == "__main__":
    test_params(combinations)
