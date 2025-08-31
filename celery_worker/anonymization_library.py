#TODO: Refactor lookup logic - dynamically assemble lookup table as entities are found by the model - kinda done, needs testing
#TODO: Refactor name normalization logic - use Petrovich/other - kinda done, needs testing

#TODO: Refactor text normalization logic completely  - top to bottom anonymization with reverse substitution in the summarizer report
#TODO: Utilize other NER models to recognize entities and add them to the lookup table

import os
import json
import hashlib
import re
import pymorphy3
from anonymization import Anonymization, AnonymizerChain, NamedEntitiesAnonymizer
from petrovich.main import Petrovich
from petrovich.enums import Case, Gender
import spacy

nlp = spacy.load("ru_core_news_lg")

morph = pymorphy3.MorphAnalyzer()
petrovich = Petrovich()

LOOKUP_PATH = "lookup.json"
LOOKUP_TABLE = {}
ENTITY_CACHE = {}

# === UTILS ===

def load_lookup_table(path=LOOKUP_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_lookup_table(table, path=LOOKUP_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)

def next_available_label(table, prefix=None):
    prefix = "ORG"
    used = [int(v.split("_")[-1].strip("]")) for v in table.values()
            if v.startswith(f"[ORG_") and v.strip("]").split("_")[-1].isdigit()]
    next_num = max(used, default=0) + 1
    return f"[ORG_{next_num}]"

def consistent_pseudonym(name, entity_type=None):
    key = f"ORG:{name.lower()}"
    if key not in ENTITY_CACHE:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:6].upper()
        ENTITY_CACHE[key] = f"[ORG_{hashed}]"
    return ENTITY_CACHE[key]

# === NORMALIZATION ===

def normalize_names(text):
    name_variants = {
        "Саша": "Александр",
        "Саня": "Александр",
        "Леша": "Алексей",
        "Лёша": "Алексей",
        "Артём": "Артем",
        "Герман": "Герман",
        "Дима": "Дмитрий",
        "Димка": "Дмитрий",
        "Паша": "Павел",
        "Макс": "Максим",
        "Маша": "Мария",
        "Машенька": "Мария",
        "Лена": "Елена",
        "Сережа": "Сергей",
        "Серёжа": "Сергей",
        "Серега": "Сергей",
        "Таня": "Татьяна",
        "Аня": "Анна",
    }
    for variant, canonical in name_variants.items():
        text = re.sub(rf'\b{re.escape(variant)}\b', canonical, text, flags=re.IGNORECASE)
    return text

def normalize_morphologically(text):
    tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
    norm_tokens = []
    for token in tokens:
        if token.strip().isalpha():
            parsed = morph.parse(token)[0]
            norm_tokens.append(parsed.normal_form)
        else:
            norm_tokens.append(token)
    return ''.join(norm_tokens)

# === NOMINATIVE NORMALIZATION ===

def normalize_to_nominative(name, entity_type=None):
    if entity_type == "PERSON":
        parts = name.split()
        if len(parts) == 3:
            lastname, firstname, middlename = parts
            try:
                gender = petrovich.gender.detect(firstname=firstname, middlename=middlename)
            except Exception:
                gender = Gender.ANDROGYNOUS
            try:
                lastname_nom = petrovich.lastname(lastname, Case.NOMINATIVE, gender)
            except Exception:
                lastname_nom = lastname
            try:
                firstname_nom = petrovich.firstname(firstname, Case.NOMINATIVE, gender)
            except Exception:
                firstname_nom = firstname
            try:
                middlename_nom = petrovich.middlename(middlename, Case.NOMINATIVE, gender)
            except Exception:
                middlename_nom = middlename
            return f"{lastname_nom} {firstname_nom} {middlename_nom}"
        elif len(parts) == 2:
            lastname, firstname = parts
            try:
                gender = petrovich.gender.detect(firstname=firstname)
            except Exception:
                gender = Gender.ANDROGYNOUS
            try:
                lastname_nom = petrovich.lastname(lastname, Case.NOMINATIVE, gender)
            except Exception:
                lastname_nom = lastname
            try:
                firstname_nom = petrovich.firstname(firstname, Case.NOMINATIVE, gender)
            except Exception:
                firstname_nom = firstname
            return f"{lastname_nom} {firstname_nom}"
        else:
            word = parts[0]
            try:
                gender = petrovich.gender.detect(firstname=word)
            except Exception:
                gender = Gender.ANDROGYNOUS
            try:
                return petrovich.firstname(word, Case.NOMINATIVE, gender)
            except Exception:
                return word
    else:
        tokens = name.split()
        norm_tokens = []
        for token in tokens:
            parsed = morph.parse(token)[0]
            norm_tokens.append(parsed.inflect({"nomn"}).word if parsed.inflect({"nomn"}) else token)
        return " ".join(norm_tokens)

# === CUSTOM PATTERNS ===

COMPANY_PATTERNS = [
    r"\b[АО]{1,2} \"[А-Яа-яA-Za-z\s]+\"",
    r"\b[А-ЯЁ][а-яё]+ Банк\b",
]

def replace_custom_patterns(text, lookup):
    for pattern in COMPANY_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            if match not in lookup:
                lookup[match] = next_available_label(lookup)
            text = text.replace(match, lookup[match])

    for key, val in lookup.items():
        text = re.sub(rf'\b{re.escape(key)}\b', val, text, flags=re.IGNORECASE)
    return text

# === PREPROCESS ===

def preprocess_text(text, lookup):
    text = normalize_names(text)
    for key in list(lookup.keys()):
        norm_key = normalize_to_nominative(key)
        if norm_key != key:
            lookup[norm_key] = lookup.pop(key)
    text = normalize_morphologically(text)
    text = replace_custom_patterns(text, lookup)
    return text

# === DYNAMIC LOOKUP UPDATE ===

def update_lookup_with_patch(patch, table):
    changed = False
    for entry in patch:
        if not isinstance(entry, dict) or "original" not in entry:
            continue
        original = entry["original"]
        if original in table:
            continue
        pseudonym = next_available_label(table)
        table[original] = pseudonym
        changed = True
    return changed

def extract_entities_spacy(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# === MAIN ANONYMIZATION ===

def anon_transcript_from_text(raw_text: str):
    lookup_table = load_lookup_table()
    preprocessed_text = preprocess_text(raw_text, lookup_table)

    # Use spaCy to extract entities
    entities = extract_entities_spacy(preprocessed_text)
    entity_map = {}  # original -> pseudonym
    reverse_patch = {}  # pseudonym -> original
    org_counter = 1
    person_counter = 1
    for ent_text, ent_label in entities:
        if ent_label == "PERSON":
            pseudonym = f"[PERSON_{person_counter}]"
            person_counter += 1
        else:
            # Use lookup table for non-human entities
            if ent_text not in lookup_table:
                pseudonym = f"[ORG_{org_counter}]"
                lookup_table[ent_text] = pseudonym
                org_counter += 1
            else:
                pseudonym = lookup_table[ent_text]
        entity_map[ent_text] = pseudonym
        reverse_patch[pseudonym] = ent_text

    # Save updated lookup table
    save_lookup_table(lookup_table)

    # Replace all entities in the text with their pseudonyms
    clean_text = preprocessed_text
    # Sort by length to avoid partial replacements
    for ent_text in sorted(entity_map, key=len, reverse=True):
        clean_text = clean_text.replace(ent_text, entity_map[ent_text])

    return clean_text, reverse_patch

# === REVERT FUNCTION ===

def revert_anon(clean_text, patch):
    anonymizer = AnonymizerChain(Anonymization('ru'))
    anonymizer.add_anonymizers(NamedEntitiesAnonymizer("ru_core_news_lg"))
    return anonymizer.revert(clean_text, patch)

if __name__ == "__main__":
    pass
    # filepath = input("Введите путь к файлу для анонимизации:\n").strip()
    # if not os.path.exists(filepath):
    #     print("Указанный файл не существует.")
    #     exit(1)

    # filename = os.path.splitext(os.path.basename(filepath))[0]
    # out_dir = "tests/anonymizer_tests/anonymization"
    # os.makedirs(out_dir, exist_ok=True)

    # clean_text, patch = anon_transcript(filepath)

    # print("\n\n---------------------PRINTING PATCH RECEIVED FROM THE ANONYMIZER---------------------\n\n")
    # print(patch)

    # # with open(f"{out_dir}/anonymized_{filename}.txt", "w", encoding="utf-8", errors="replace") as file:
    # #     file.write(clean_text)

    # revert_text = revert_anon(clean_text, patch)

    # # with open(f"{out_dir}/reverted_text_{filename}.txt", "w", encoding="utf-8", errors="replace") as file:
    # #     file.write(revert_text)