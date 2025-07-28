# import hashlib
# import json
# from deeppavlov import build_model, configs
# from typing import List, Tuple

# print("---------------Loading NER and coreference models---------------")
# ner_model = build_model(configs.ner.ner_rus_bert, download=True)
# coref_model = build_model(configs.coref.rus_coref, download=True)

# with open("lookup.json", encoding="utf-8") as f:
#     lookup = json.load(f)

# pseudonym_map = {}

# def run_coreference(text: str) -> str:
#     try:
#         resolved = coref_model([text])[0]
#         return resolved
#     except Exception as e:
#         print(f"[Coreference Error] {e}")
#         return text

# def extract_entities(text: str) -> List[Tuple[str, str]]:
#     words, tags = ner_model([text])
#     entities = []
#     current_entity = ""
#     current_type = None

#     for word, tag in zip(words[0], tags[0]):
#         if tag.startswith("B-"):
#             if current_entity:
#                 entities.append((current_entity.strip(), current_type))
#             current_entity = word
#             current_type = tag[2:]
#         elif tag.startswith("I-") and current_type:
#             current_entity += " " + word
#         else:
#             if current_entity:
#                 entities.append((current_entity.strip(), current_type))
#                 current_entity = ""
#                 current_type = None

#     if current_entity:
#         entities.append((current_entity.strip(), current_type))

#     return entities

# def consistent_pseudonym(name: str, entity_type: str) -> str:
#     key = f"{entity_type}:{name.lower()}"
#     if key not in pseudonym_map:
#         hash_digest = hashlib.sha1(key.encode()).hexdigest()[:8]
#         pseudonym_map[key] = f"[{entity_type.upper()}_{hash_digest}]"
#     return pseudonym_map[key]


# def apply_lookup(text: str) -> str:
#     for name, replacement in lookup.items():
#         text = text.replace(name, replacement)
#     return text



# def anonymize_text(text: str) -> str:
#     print("Running coreference resolution...")
#     resolved = run_coreference(text)

#     print("Extracting named entities...")
#     entities = extract_entities(resolved)

#     print("Replacing with consistent pseudonyms...")
#     for entity, entity_type in sorted(entities, key=lambda x: -len(x[0])): 
#         pseudonym = consistent_pseudonym(entity, entity_type)
#         resolved = resolved.replace(entity, pseudonym)

#     print("Applying manual lookup replacements...")
#     resolved = apply_lookup(resolved)

#     return resolved


# def anon_transcript(filepath: str):
#     with open(filepath, "r", encoding="utf-8") as file:
#         text = file.read()

#     clean_text = anonymize_text(text)

#     with open("anonymized_output.txt", "w", encoding="utf-8") as f:
#         f.write(clean_text)

#     print("\nAnonymized text written to anonymized_output.txt\n")
#     return clean_text


# if __name__ == "__main__":
#     filepath = input("Enter path to transcript file:\n").strip()
#     anon_transcript(filepath)