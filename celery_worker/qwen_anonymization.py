import os
import requests
from tokenCounter import count_tokens
import re
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def remove_tagged_text(text:str, tag:str) -> str:
        """
        Remove parts of text enclosed in a specific tag.
        """
        pattern = re.compile(r"<" + tag + r">.*?</" + tag + r">", re.DOTALL)
        return re.sub(pattern, "", text)

class Anonymization_Mapping(BaseModel):
    mappings: dict[str, str]

def get_anonymization_mapping(text:str, model_name:str):
    """
    Generates anonymization mapping from the provided text.
    """
    prompt = f"""Ты - специалист по анонимизации коммерческой информации. Твоя задача - заменить коммерческую информацию в транскриптах встреч на стандартизированные псевдонимы и предоставить полную карту замен, которые были сделаны. Четко следуй следующим правилам для анонимизации данных:

ПРАВИЛА АНОНИМИЗАЦИИ:

1. ИМЕНА ЛЮДЕЙ:
   - Заменяй на формат: [PERSON_X] где X - порядковый номер
   - Пример: "Иван Петров" → [PERSON_1], "Мария Сидорова" → [PERSON_2]

2. НАЗВАНИЯ КОМПАНИЙ:
   - Заменяй на формат: [COMPANY_X] где X - порядковый номер
   - Пример: "ООО Рога и Копыта" → [COMPANY_1], "АО Светлое будущее" → [COMPANY_2]

3. НАЗВАНИЯ ПРОЕКТОВ/ПРОДУКТОВ:
   - Заменяй на формат: [PROJECT_X] где X - порядковый номер
   - Пример: "проект Альфа" → [PROJECT_1], "система CRM" → [PROJECT_2]

4. КЛИЕНТЫ/ПАРТНЕРЫ:
   - Заменяй на формат: [CLIENT_X] где X - порядковый номер
   - Пример: "наш клиент Сбербанк" → "наш клиент [CLIENT_1]"

5. ФИНАНСОВЫЕ ДАННЫЕ:
   - Конкретные суммы заменяй на: [AMOUNT_X]
   - Пример: "1 миллион рублей" → "[AMOUNT_1]"

ВАЖНЫЕ ТРЕБОВАНИЯ:
- НЕ заменяй общие должности (директор, менеджер, разработчик)
- НЕ заменяй общие термины (встреча, проект, задача)
- Веди подсчет замен по категориям

Транскрипт для анонимизации:
{text}
   """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":0.5,
            "num_predict": 2000,
            "num_ctx": count_tokens(text=text) + 2000,
        },
        "format": Anonymization_Mapping.model_json_schema(),
        "think": True,
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        print("\n======RAW RESPONSE======\n")
        print(response_json)
        print("\n")

        anonymization_mapping = Anonymization_Mapping.model_validate_json(response_json.get("response").strip())

        # anonymization_mapping = remove_tagged_text(text=anonymization_mapping, tag="think")

        return anonymization_mapping
    except Exception as e:
        return f"[ERROR]: {e}"



def anonymize_text(text, model_name):
    prompt = f"""Ты - специалист по анонимизации деловых документов. Твоя задача - заменить коммерческую информацию в транскриптах встреч на стандартизированные псевдонимы, сохраняя при этом структуру и смысл текста. Выдай предоставленный текст ПОЛНОСТЬЮ без потери какой-либо информации с замененной коммерческой информацией.

ПРАВИЛА АНОНИМИЗАЦИИ:

1. ИМЕНА ЛЮДЕЙ:
   - Заменяй на формат: [PERSON_X] где X - порядковый номер
   - Пример: "Иван Петров" → [PERSON_1], "Мария Сидорова" → [PERSON_2]

2. НАЗВАНИЯ КОМПАНИЙ:
   - Заменяй на формат: [COMPANY_X] где X - порядковый номер
   - Пример: "ООО Рога и Копыта" → [COMPANY_1], "АО Светлое будущее" → [COMPANY_2]

3. НАЗВАНИЯ ПРОЕКТОВ/ПРОДУКТОВ:
   - Заменяй на формат: [PROJECT_X] где X - порядковый номер
   - Пример: "проект Альфа" → [PROJECT_1], "система CRM" → [PROJECT_2]

4. КЛИЕНТЫ/ПАРТНЕРЫ:
   - Заменяй на формат: [CLIENT_X] где X - порядковый номер
   - Пример: "наш клиент Сбербанк" → "наш клиент [CLIENT_1]"

5. ФИНАНСОВЫЕ ДАННЫЕ:
   - Конкретные суммы заменяй на: [AMOUNT_X]
   - Пример: "1 миллион рублей" → "[AMOUNT_1]"

ВАЖНЫЕ ТРЕБОВАНИЯ:
- Сохраняй контекст и грамматическую структуру предложений
- НЕ заменяй общие должности (директор, менеджер, разработчик)
- НЕ заменяй общие термины (встреча, проект, задача)
- Веди подсчет замен по категориям
- В конце текста добавь MAPPING секцию со всеми заменами

ФОРМАТ ОТВЕТА:
1. ПОЛНЫЙ Анонимизированный текст
2. === MAPPING ===
3. Список всех замен в формате: [ПСЕВДОНИМ] = Оригинальный текст

Пример mapping:
=== MAPPING ===
[PERSON_1] = Иван Петров
[PERSON_2] = Мария Сидорова  
[COMPANY_1] = ООО Рога и Копыта
[PROJECT_1] = проект Альфа
[AMOUNT_1] = 500 тысяч рублей

Текст для анонимизации:
{text}"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":0.6,
            "num_ctx": count_tokens(text=text) + 5000,
            "num_predict": count_tokens(text=text) + 1000
        },
        "think": False,
    }
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()

        anonymized_text = response_json.get("response").strip()

        return anonymized_text
    except Exception as e:
        return "[SUMMARY_FAILED]"

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
    