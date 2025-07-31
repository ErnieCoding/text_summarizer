#TODO: Add text preprocessing after mappings for case normalization

#TODO: Be able to create 20 anonymization mappings - compare and send to the chat. Goal is to be able to standardize mappings creation.

#TODO: Go through multiple rounds of prompts to create the mappings. Round 1: prompt with existing mappings and update it if new entities have been identified, Round 2: anonymize text algorithmically, Round 3: depending on the end result - create another mapping or compare the already existing mapping to what the llm has produced
import os
import requests
from tokenCounter import count_tokens
import re
from pydantic import BaseModel
import json

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def remove_tagged_text(text:str, tag:str) -> str:
    """
    Remove parts of text enclosed in a specific tag.
    """
    pattern = re.compile(r"<" + tag + r">.*?</" + tag + r">", re.DOTALL)
    return re.sub(pattern, "", text)

def get_anonymization_mapping(text: str, model_name: str) -> dict[str, str] | str:
    """
    Generates anonymization mapping from the provided text using LLM.
    Returns a dict with mappings or an error string.
    """
    prompt = f"""Ты — эксперт по анонимизации коммерческой информации. Твоя задача — проанализировать текст транскрипта и заменить все конфиденциальные данные на стандартизированные псевдонимы. Верни **строго** один **валидный JSON-объект** без каких-либо пояснений, комментариев или дополнительного текста. Формат **всегда** должен быть **одинаковым**:

{{
  "mappings": {{
    "Оригинальное_значение_1": "Псевдоним_1",
    "Оригинальное_значение_2": "Псевдоним_2"
  }}
}}

Ты обязан:
- Вернуть **только** JSON с ключом верхнего уровня **"mappings"**.
- Не писать ни одного слова до или после JSON (никаких пояснений, комментариев, пустых строк, `json:` и т.п.).
- Соблюдать **точный** синтаксис JSON (кавычки, запятые, формат вложенности).
- Сохранять **порядковую нумерацию** псевдонимов по каждой категории отдельно, начиная с 1.

ПРАВИЛА АНОНИМИЗАЦИИ (строго соблюдай):

1. ИМЕНА ЛЮДЕЙ → [PERSON_X]
   (например: "Иван Петров" → "[PERSON_1]")

2. НАЗВАНИЯ КОМПАНИЙ → [COMPANY_X]
   (например: "АО Светлое будущее" → "[COMPANY_1]")

3. НАЗВАНИЯ ПРОЕКТОВ И ПРОДУКТОВ → [PROJECT_X]

4. КЛИЕНТЫ, ПАРТНЁРЫ → [CLIENT_X]

5. ФИНАНСОВЫЕ ДАННЫЕ (суммы, бюджеты, цены) → [AMOUNT_X]
   (например: "12 миллионов рублей" → "[AMOUNT_1]")

НЕ АНОНИМИЗИРУЙ:
- Общие должности (директор, менеджер, инженер и т.п.)
- Общие термины (встреча, проект, задача и т.п.)

ПРИМЕР ВЫВОДА:

{{
  "mappings": {{
    "Иван Петров": "[PERSON_1]",
    "АО Светлое будущее": "[COMPANY_1]",
    "Проект Альфа": "[PROJECT_1]",
    "ООО Рога и Копыта": "[CLIENT_1]",
    "12 миллионов рублей": "[AMOUNT_1]"
  }}
}}

ВАЖНО: Верни результат **строго в этом формате**. Ни одного лишнего символа.

Вот транскрипт:

{text}
"""


    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 3000,
            "num_ctx": count_tokens(text=text) + 1000,
        },
        #"stop": ["}\n}"],
        "think": False
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()

        response_json = response.json()

        print("\n======RAW JSON RESPONSE======\n")
        print(response_json)
        print("\n")

        raw_text = response.json().get("response", "").strip()

        if not raw_text:
            return "[ERROR]: Empty response from model."
        
        print("\n======RAW TEXT RESPONSE======\n")
        print(raw_text)
        print("\n")

        match = re.search(r'\{\s*"mappings"\s*:\s*\{.*?\}\s*\}', raw_text, re.DOTALL)
        if not match:
            return "[ERROR]: No valid JSON object with 'mappings' found."

        json_str = match.group(0)

        parsed = json.loads(json_str)

        if "mappings" not in parsed:
            return "[ERROR]: JSON does not contain 'mappings' key."
        
        safe_model = re.sub(r'[^\w.-]+', '_', model_name)
        filename = f"tests/anonymizer_tests/qwen/{safe_model}_mappings_no_preprocess.txt"
        with open(filename, "w+", encoding="utf-8", newline="") as file:
            file.write(f"МОДЕЛЬ: {model_name}\n\n")
            file.write("ДАННЫЕ (JSON):\n")
            file.write(json.dumps(parsed, ensure_ascii=False, indent=2))
            file.write("\n")

        return parsed["mappings"]

    except json.JSONDecodeError as e:
        print(f"[JSON DECODE ERROR]: {e}")
        print("===============================")
        print(f"RAW TEXT:\n{raw_text}\n")
        return f"[JSON ERROR]: {e}"
    except Exception as e:
        return f"[ERROR]: {e}"


if __name__ == "__main__":
    filepath = "transcripts/Командос23-12.txt"
    print(f"\nФАЙЛ: {filepath}\n")
    with open(f"{filepath}", "r", encoding="utf-8") as file:
        orginal_text = file.read()

    model_name = input("Введите название модели для анонимизации:\n").strip()

    print("\n============Starting anonymization pipeline============\n")
    anon_mapping = get_anonymization_mapping(orginal_text, model_name)

    print(anon_mapping)

    # print("\n============SAVING ANONYMIZED TEXT============\n")
    # anon_filename = filepath.split("/").split(".")[0]
    # with open(f"tests/anonymizer_tests/qwen/qwen_anon_{anon_filename}.txt", "w+", encoding="utf-8", errors="replace") as file:
    #     file.write(anon_mapping)
    