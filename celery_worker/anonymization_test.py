import hashlib
from anonymization import Anonymization, AnonymizerChain, NamedEntitiesAnonymizer

ENTITY_CACHE = {}

def consistent_pseudonym(name, entity_type):
    key = f"{entity_type}:{name.lower()}"
    if key not in ENTITY_CACHE:
        ENTITY_CACHE[key] = f"[{entity_type.upper()}_{len(ENTITY_CACHE)+1}]"
    return ENTITY_CACHE[key]

def normalize_names(text):
    name_variants = {
        "Саша": "Александр",
        "Саня": "Александр",
        "Герман": "Герман",
        "Альфа": "Альфа Банк",
        "Ольфу Банк": "Альфа Банк", 
    }
    for variant, canonical in name_variants.items():
        text = text.replace(variant, canonical)
    return text

def anon_transcript(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = normalize_names(file.read())

    anonymizer = AnonymizerChain(Anonymization('ru'))
    anonymizer.add_anonymizers(NamedEntitiesAnonymizer("ru_core_news_lg"))
    clean_text, patch = anonymizer.pseudonymize(text)

    new_patch = {}
    for original, pseudonym in patch.items():
        entity_type = pseudonym.split("_")[0].strip("[]")
        new_patch[original] = consistent_pseudonym(original, entity_type)
        clean_text = clean_text.replace(pseudonym, new_patch[original])

    return clean_text, new_patch

def revert_anon(clean_text, patch):
    anonymizer = AnonymizerChain(Anonymization('ru'))
    anonymizer.add_anonymizers(NamedEntitiesAnonymizer("ru_core_news_lg"))

    return anonymizer.revert(clean_text, patch)

if __name__ == "__main__":
    filepath = input("Input the file you want to anonymize:\n")

    clean_text, patch = anon_transcript(filepath)

    print(clean_text + "\n\n\n\n")
    with open("tests/anonymizer_tests/anonymized_test1.txt", "w+", encoding="utf-8", errors="replace") as file:
        file.write(clean_text)

    revert_text = revert_anon(clean_text, patch)
    print(revert_text)
    with open("tests/anonymizer_tests/reverted_text.txt", "w+", encoding="utf-8", errors="replace") as file:
        file.write(revert_text)