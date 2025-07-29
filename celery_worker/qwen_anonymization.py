#TODO: Be able to create 20 anonymization mappings - compare and send to the chat. Goal is to be able to standardize mappings creation.

#TODO: Add text preprocessing after mappings for case normalization

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
    prompt = f"""Ты - специалист по анонимизации коммерческой информации. Твоя задача - заменить коммерческую информацию в транскриптах встреч на стандартизированные псевдонимы и предоставить полную карту замен, которые были сделаны. Верни **ТОЛЬКО** JSON-объект формата:

{{
  "mappings": {{
    "Оригинальное_значение_1": "Псевдоним_1",
    "Оригинальное_значение_2": "Псевдоним_2"
  }}
}}

Пример:
{{
  "mappings": {{
    "Иван Петров": "[PERSON_1]",
    "АО Светлое будущее": "[COMPANY_1]"
  }}
}}

Четко следуй следующим правилам для анонимизации данных:

ПРАВИЛА АНОНИМИЗАЦИИ:

1. ИМЕНА ЛЮДЕЙ:
   - Заменяй на формат: [PERSON_X] где X - порядковый номер

2. НАЗВАНИЯ КОМПАНИЙ:
   - Заменяй на формат: [COMPANY_X] где X - порядковый номер

3. НАЗВАНИЯ ПРОЕКТОВ/ПРОДУКТОВ:
   - Заменяй на формат: [PROJECT_X] где X - порядковый номер

4. КЛИЕНТЫ/ПАРТНЕРЫ:
   - Заменяй на формат: [CLIENT_X] где X - порядковый номер

5. ФИНАНСОВЫЕ ДАННЫЕ:
   - Конкретные суммы заменяй на: [AMOUNT_X]

ВАЖНЫЕ ТРЕБОВАНИЯ:
- НЕ заменяй общие должности (директор, менеджер, разработчик)
- НЕ заменяй общие термины (встреча, проект, задача)
- Веди подсчет замен по категориям

Не добавляй комментарии, текст до или после JSON. Верни только валидный JSON. Вот транскрипт:

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
        "stop": ["}\n}"],
        "think": True
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()

        response_text = response_json.get("response", "").strip()
        if not response_text:
            raise ValueError("Empty response received from model.")

        last_brace = response_text.rfind("}")
        if last_brace != -1:
            response_text = response_text[:last_brace + 1]

        print("\n======RAW RESPONSE======\n")
        print(response_text)
        print("\n")

        parsed = json.loads(response_text)
        return parsed.get("mappings", {})

    except json.JSONDecodeError as e:
        print(f"[JSON DECODE ERROR]: {e}")
        print("===============================")
        print(f"RAW TEXT:\n{response_text}\n")
        return f"[JSON ERROR]: {e}"
    except Exception as e:
        return f"[ERROR]: {e}"

if __name__ == "__main__":
    filepath = input("Введите путь к файлу:\n")
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
    