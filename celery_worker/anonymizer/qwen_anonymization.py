#TODO: Add text preprocessing after mappings for case normalization

#TODO: Попробовать промпты Алекся для создания карты + саммари

#TODO: Go through multiple rounds of prompts to create the mappings. Round 1: prompt with existing mappings and update it if new entities have been identified, Round 2: anonymize text algorithmically, Round 3: depending on the end result - create another mapping or compare the already existing mapping to what the llm has produced
import os
import requests
from tokenCounter import count_tokens
import re
from pydantic import BaseModel
import json
import whisper, torch

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
    prompt = f"""ЗАДАЧА: Создание карты замен для анонимизации транскрипта бизнес-встречи

КРИТИЧЕСКИ ВАЖНО: Твоя задача - выявить ВСЕ чувствительные данные без исключений.

ОБЯЗАТЕЛЬНЫЕ ПРИНЦИПЫ:
1. ПОЛНОТА: Найди каждое упоминание чувствительных данных, включая косвенные ссылки
2. КОНСИСТЕНТНОСТЬ: Одинаковые сущности = одинаковые замены по всему тексту
3. БЕЗОПАСНОСТЬ: При сомнениях выбирай анонимизацию

КАТЕГОРИИ ДЛЯ ПОИСКА (приоритетные):

🔴 КОМПАНИИ И ПРОДУКТЫ (ВЫСОКИЙ ПРИОРИТЕТ):
- Названия компаний (включая сокращения, аббревиатуры)  
- Бренды и торговые марки
- Названия продуктов и сервисов
- Внутренние кодовые названия проектов
- Названия систем и платформ

🔴 ПЕРСОНАЛЬНЫЕ ДАННЫЕ (ВЫСОКИЙ ПРИОРИТЕТ):
- ФИО участников (полные и частичные)
- Должности с названиями компаний
- Email адреса и телефоны
- Внутренние логины и аккаунты

🔴 ГЕОГРАФИЯ:
- Города, регионы, страны
- Адреса офисов и подразделений
- Названия локаций и площадок

🔴 ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ (ВЫСОКИЙ ПРИОРИТЕТ):
- Модели оборудования и серийные номера
- IP-адреса, домены, URL
- Версии ПО и конфигурации
- Технические характеристики

🔴 ФИНАНСЫ:
- Конкретные суммы и бюджеты
- Номера договоров
- Коммерческие условия

🔴 ВРЕМЕННЫЕ ДАННЫЕ:
- Конкретные даты (оставляй только относительные: "на прошлой неделе")
- Сроки проектов и дедлайны

СПЕЦИАЛЬНЫЕ ИНСТРУКЦИИ ДЛЯ АВТОТРАНСКРИПЦИИ:

⚠️ ВНИМАНИЕ НА ОШИБКИ РАСПОЗНАВАНИЯ:
- Названия компаний могут быть искажены: "Майкрософт", "Гугл", "Сбербанк"
- Продукты могут звучать неправильно: "Офис 365", "Виндовс", "Андроид"  
- Технические термины могут быть неточными
- Имена людей часто распознаются неверно

ФОРМАТ ЗАМЕН:
- Используй четкую систему: КОМПАНИЯ_1, ПРОДУКТ_1, УЧАСТНИК_1
- Сохраняй числовую последовательность в рамках категории
- НЕ указывай тип деятельности в замене (НЕ "Банк_1", а "КОМПАНИЯ_1")

ФОРМАТ ВЫВОДА:

=== КАРТА ЗАМЕН ===

ПЕРСОНАЛЬНЫЕ_ДАННЫЕ:
[Исходное] → [Замена]
Иван Петров → УЧАСТНИК_1  
Мария Сидорова → УЧАСТНИК_2
ivan.petrov@company.com → EMAIL_1

КОМПАНИИ_И_ПРОДУКТЫ:
[Исходное] → [Замена]
Microsoft → КОМПАНИЯ_1
Office 365 → ПРОДУКТ_1
Windows Server → ПРОДУКТ_2

ГЕОГРАФИЯ:
[Исходное] → [Замена]
Москва → ГОРОД_1
Санкт-Петербург → ГОРОД_2

ТЕХНИЧЕСКАЯ_ИНФОРМАЦИЯ:
[Исходное] → [Замена]
Dell R740 → ОБОРУДОВАНИЕ_1
192.168.1.1 → IP_АДРЕС_1

ФИНАНСЫ:
[Исходное] → [Замена]
1500000 рублей → СУММА_1
Договор №12/2024 → ДОГОВОР_1

ВРЕМЕННЫЕ_ДАННЫЕ:
[Исходное] → [Замена]
15 марта 2024 → ДАТА_1
к концу квартала → СРОК_1

=== КОНЕЦ КАРТЫ ЗАМЕН ===

КОНТРОЛЬ КАЧЕСТВА:
✅ Проверь каждую категорию дважды
✅ Убедись, что одинаковые сущности имеют одинаковые замены
✅ Найди все варианты написания названий (сокращения, ошибки транскрипции)
✅ Обрати особое внимание на бренды и продукты - их пропускают чаще всего

Текст транскрипта для анализа:

{text}
"""


    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.85,
            "repeat_penalty": 1.15,
            "num_predict": 3500,
            "num_ctx": min(count_tokens(prompt) + 3000, 32768),
        },
        "think": False
    }

    try:
        print("\n\n=====STARTING MAPPING GENERATION======\n")
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()

        response_text = response.json().get("response", "").strip()

        if not response_text:
            return "[ERROR]: Empty response from model."
        
        print("\n======MAPPING GENERATION RESPONSE======\n")
        print(response_text)
        print("\n\n\n")

        print("MAPPING SAVING IN PROGRESS....\n")
        safe_model = re.sub(r'[^\w.-]+', '_', model_name)
        filename = f"tests/anonymizer_tests/qwen/{safe_model}_mappings(CLAUDE_PROMPTS).txt"
        with open(filename, "w+", encoding="utf-8", newline="") as file:
         file.write(f"МОДЕЛЬ: {model_name}\n\n")
         file.write("ОТВЕТ:\n")
         file.write(response_text)
         file.write("\n")


        print("\n!DONE SAVING MAPPINGS!\n")
        return response_text
    except Exception as e:
        return f"[ERROR]: {e}"

def anonymize_transcript(text: str, mappings: str, model_name: str = "qwen3:30b"):
    prompt = f"""ЗАДАЧА: Применить карту замен и очистить транскрипт с умеренным сокращением

ВАЖНО: Цель - сократить на 40-50% (не 70%!), сохранив ключевую информацию и контекст.

ЭТАП 1 - АНОНИМИЗАЦИЯ:

СТРОГИЕ ПРАВИЛА ЗАМЕНЫ:
1. Найди в карте замен КАЖДУЮ строку формата: [Исходное] → [Замена]
2. Замени ВСЕ вхождения [Исходного] на [Замена] по всему тексту
3. Учитывай регистр и словоформы: "Microsoft" и "майкрософт" = КОМПАНИЯ_1
4. Сохраняй контекст: смысл фраз не должен теряться
5. При отсутствии в карте - оставляй как есть, НЕ придумывай новые замены

ЭТАП 2 - РАЗМЕТКА УЧАСТНИКОВ:
- Первый говорящий = УЧАСТНИК_1  
- Второй говорящий = УЧАСТНИК_2
- Модератор/ведущий = МОДЕРАТОР (если есть)
- Сохрани структуру диалога с именами ролей

ЭТАП 3 - ОЧИСТКА (УМЕРЕННАЯ):

УБРАТЬ ОБЯЗАТЕЛЬНО:
- Междометья: "эээ", "ммм", "хм", "ааа"
- Повторы: "это это важно" → "это важно" 
- Техническое: "меня слышно?", "связь плохая"
- Слова-паразиты: "короче", "типа", "как бы", "в общем-то"

СОКРАТИТЬ ОСТОРОЖНО:
- "Я хотел бы сказать, что нам стоит рассмотреть" → "Предлагаю рассмотреть"
- "Возможно, нам следует подумать о том, что" → "Стоит рассмотреть"
- "Если я правильно понимаю, то получается" → "То есть"

СОХРАНИТЬ ОБЯЗАТЕЛЬНО:
- Все решения и выводы
- Конкретные действия и ответственных  
- Важные вопросы и ответы
- Аргументы и обоснования
- Технические детали (в анонимизированном виде)
- Временные рамки и дедлайны

ЭТАП 4 - СТРУКТУРИРОВАНИЕ:

ФОРМАТ ВЫВОДА:

=== АНОНИМИЗИРОВАННЫЙ ТРАНСКРИПТ ===

УЧАСТНИКИ:
- УЧАСТНИК_1: [анонимизированная роль]
- УЧАСТНИК_2: [анонимизированная роль]  
- МОДЕРАТОР: [анонимизированная роль]

ОСНОВНОЕ СОДЕРЖАНИЕ:

**[время/тема]**
УЧАСТНИК_1: [очищенная речь]
УЧАСТНИК_2: [очищенная речь]

**[следующая тема]**
[продолжение диалога...]

КЛЮЧЕВЫЕ РЕШЕНИЯ:
1. [решение в анонимизированном виде]
2. [решение в анонимизированном виде]

ДЕЙСТВИЯ:
| Что | Кто | Когда |
|-----|-----|-------|
| [действие] | [роль] | [срок] |

ВОПРОСЫ К ПРОРАБОТКЕ:
- [вопрос 1]
- [вопрос 2]

=== СТАТИСТИКА ОБРАБОТКИ ===
Исходная длина: [слов]
Финальная длина: [слов]  
Сокращение: [%] (цель: 40-50%)
Количество замен: [число]

КОНТРОЛЬ КАЧЕСТВА:
✅ Все замены из карты применены
✅ Сохранен смысл и контекст  
✅ Убраны шумы, но оставлена суть
✅ Структура диалога понятна

ВХОДНЫЕ ДАННЫЕ:

КАРТА ЗАМЕН:
{mappings}

ИСХОДНЫЙ ТРАНСКРИПТ:
{text}
"""
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 9000,
            "num_ctx": min(count_tokens(prompt) + 5000, 32768),
        },
        "think": False
    }

    try:
        print("=====STARTING TRANSCRIPT ANONYMIZATION=====")
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        response.raise_for_status()

        response_text = response.json().get("response", "").strip()

        if not response_text:
            return "[ERROR]: Empty response from model."
        
        print("\n======TRANSCRIPT ANONYMIZATION RESPONSE======\n")
        print(response_text)
        print("\n\n\n")
        
        print("ANON TRANSCRIPT SAVING IN PROGRESS....\n")
        safe_model = re.sub(r'[^\w.-]+', '_', model_name)
        filename = f"tests/anonymizer_tests/qwen/{safe_model}_anontranscript(CLAUDE_PROMPTS).txt"
        with open(filename, "w+", encoding="utf-8", newline="") as file:
         file.write(f"МОДЕЛЬ: {model_name}\n\n")
         file.write("ОТВЕТ:\n")
         file.write(response_text)
         file.write("\n")

        print("!DONE!")
        return response_text
    except Exception as e:
        return f"[ERROR]: {e}"

def transcribe_meeting():
   filepath = "new_meeting.mp3"

   print(f"Saved uploaded file to: {filepath}")
   print("STARTING TRANSCRIPTION")

   print(f"\nPyTorch version: {torch.__version__}\n")
   print(f"\nCUDA version: {torch.version.cuda}\n")
   print(f"\nCUDA available: {torch.cuda.is_available()}\n")
   print(f"\ncuDNN enabled: {torch.backends.cudnn.enabled}\n")

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = whisper.load_model("large").to(device)

   result = model.transcribe(filepath, fp16 = (device == "cuda"))

   output_filename = os.path.splitext(os.path.basename(filepath))[0]
   output_path = os.path.join("transcripts", f"{output_filename}.txt")
   with open(output_path, "w+", encoding="utf-8") as file:
      file.write(result["text"])
   
   return output_path

if __name__ == "__main__":
   filepath = "transcripts/Диоризированные транскрипты/new_meeting.txt"

   # option = int(input("Choose file option:\n 1. Командос17-12.txt \t 2. Командос23-12.txt\n"))
   # if option == 2:
   #    filepath = "transcripts/Командос23-12.txt"
   # elif option == 1:
   #    filepath = "transcripts/Командос17-12.txt"

   print(f"\nФАЙЛ ВЫБРАН: {filepath}\n")
   with open(f"{filepath}", "r", encoding="utf-8") as file:
      orginal_text = file.read()

   mapping_model_name = "qwen3:30b"

   print("\n============Starting anonymization pipeline============\n")
   anon_mapping = get_anonymization_mapping(orginal_text, mapping_model_name)

   anon_model_name = "qwen3:30b"
   anonymize_transcript(orginal_text, anon_mapping, model_name=anon_model_name)
    