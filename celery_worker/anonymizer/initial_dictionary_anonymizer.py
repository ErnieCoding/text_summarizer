from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import re
import os
import requests

from natasha import (
    Segmenter,
    MorphVocab,
    Doc,
    NewsEmbedding,
    NewsNERTagger,
    DatesExtractor,
    MoneyExtractor
)
import pymorphy2
from petrovich.main import Petrovich
from petrovich.enums import Case, Gender


# =========Константы=========
TAG_ORDER = [
    "PERSON", "ORG", "LOC", "DATE", "MONEY", "PERCENT", "NUMBER",
    "TECH", "PRODUCT", "PLATFORM"
]

DEFAULT_TECH_LEXICON = {
    "TECH": ["AI", "ML", "NLP", "LLM", "API", "SDK", "SaaS", "ERP", "CRM", "Python", "Java", "Go", "Rust"],
    "PRODUCT": ["Salesforce", "SAP", "1C", "Bitrix24", "GitLab"],
    "PLATFORM": ["AWS", "GCP", "Azure", "Yandex Cloud"]
}


@dataclass
class QwenConfig:
    host: str = "http://localhost:11434"
    endpoint: str = "/api/generate"
    model: str = "qwen3:30b"
    temperature: float = 0.4
    max_tokens: int = 3000


# =========Qwen клиент=========
class QwenOllamaClient:
    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.host + self.cfg.endpoint
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def extract_domain_terms(self, text: str, already_found: Dict[str, str]) -> Dict[str, str]:
        system = (
            "Ты помогаешь анонимизировать протоколы. "
            "Выдели только доменные сущности (TECH, PRODUCT, PLATFORM). "
            "Игнорируй имена, даты, деньги, числа. Верни только JSON."
        )

        already = json.dumps(already_found, ensure_ascii=False)
        user = f"""Текст:
<<<
{text}
>>>

Уже найденные замены:
{already}
Верни JSON: {{"mappings": {{"Qwen": "[TECH_01]"}}}}"""

        prompt = f"<s>[SYSTEM]{system}[/SYSTEM]\n[USER]{user}[/USER]"

        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            }
        }

        try:
            raw = self._post(payload).get("response", "").strip()
            parsed = json.loads(raw)
            return parsed.get("mappings", {})
        except Exception:
            return {}


# ========= EntitySpan =========
@dataclass
class EntitySpan:
    start: int
    end: int
    surface: str
    etype: str


# ========= Анонимизатор =========
class HybridAnonymizer:
    def __init__(
        self,
        qwen: Optional[QwenOllamaClient] = None,
        tech_lexicon: Optional[Dict[str, List[str]]] = None,
        initial_dict_path: str = "initial_dict.json"
    ):
        # Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner = NewsNERTagger(self.emb)
        self.dates = DatesExtractor(self.morph_vocab)
        self.money = MoneyExtractor(self.morph_vocab)

        self.morph = pymorphy2.MorphAnalyzer()
        self.petrovich = Petrovich()
        self.qwen = qwen
        self.tech_lexicon = tech_lexicon or DEFAULT_TECH_LEXICON
        self.counters = defaultdict(int)

        # Загрузка словаря
        self.mapping = self._load_initial_dict(initial_dict_path)
        self.initial_dict_path = initial_dict_path

    def _load_initial_dict(self, path: str) -> Dict[str, str]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_mapping(self):
        with open(self.initial_dict_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)

    def _new_tag(self, etype: str) -> str:
        self.counters[etype] += 1
        return f"[{etype}_{self.counters[etype]:02d}]"

    def _remember(self, surface: str, etype: str):
        if surface not in self.mapping:
            self.mapping[surface] = self._new_tag(etype)

    def _extract_natasha_entities(self, text: str) -> List[EntitySpan]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner)
        spans = []

        for span in doc.spans:
            span.normalize(self.morph_vocab)
            if span.type == "PER":
                spans.append(EntitySpan(span.start, span.stop, span.text, "PERSON"))
            elif span.type == "ORG":
                spans.append(EntitySpan(span.start, span.stop, span.text, "ORG"))
            elif span.type == "LOC":
                spans.append(EntitySpan(span.start, span.stop, span.text, "LOC"))

        for m in self.dates(text):
            spans.append(EntitySpan(m.start, m.stop, text[m.start:m.stop], "DATE"))

        for m in self.money(text):
            spans.append(EntitySpan(m.start, m.stop, text[m.start:m.stop], "MONEY"))

        percent_regex = re.compile(r"(?<!\S)(\d+[.,]?\d*)\s*%")
        for mm in percent_regex.finditer(text):
            spans.append(EntitySpan(mm.start(1), mm.end(0), mm.group(0), "PERCENT"))

        for mm in re.finditer(r'\b\d+[.,]?\d*\b', text):
            val = mm.group(0)
            if not re.fullmatch(r'\d+[.,]?\d*%', val):
                spans.append(EntitySpan(mm.start(), mm.end(), val, "NUMBER"))

        return spans

    def _extract_lexicon_terms(self, text: str) -> List[EntitySpan]:
        spans = []
        for etype, words in self.tech_lexicon.items():
            for w in words:
                for m in re.finditer(rf"(?<!\w){re.escape(w)}(?!\w)", text, re.IGNORECASE):
                    spans.append(EntitySpan(m.start(), m.end(), m.group(0), etype))
        return spans

    def _add_case_variants(self, forms: List[str]) -> List[str]:
        """Добавляет к списку словоформ варианты с заглавной и строчной буквы."""
        extended = set()
        for form in forms:
            extended.add(form)
            extended.add(form.lower())
            extended.add(form.capitalize())
        return list(extended)

    def _generate_person_forms(self, name: str) -> List[str]:
        parts = name.split()
        if not parts:
            return [name]

        lastname = parts[0] if len(parts) > 0 else None
        firstname = parts[1] if len(parts) > 1 else None
        middlename = parts[2] if len(parts) > 2 else None

        gender = Gender.MALE
        if middlename:
            p = self.morph.parse(middlename)[0]
            if "femn" in p.tag:
                gender = Gender.FEMALE

        forms = set()
        for case in [Case.GENITIVE, Case.DATIVE, Case.ACCUSATIVE, Case.INSTRUMENTAL, Case.PREPOSITIONAL]:
            try:
                ln = self.petrovich.lastname(lastname, case=case, gender=gender) if lastname else ""
                fn = self.petrovich.firstname(firstname, case=case, gender=gender) if firstname else ""
                mn = self.petrovich.middlename(middlename, case=case, gender=gender) if middlename else ""
                forms.add(" ".join(filter(None, [ln, fn, mn])))
            except Exception:
                pass

        forms.add(name)
        return self._add_case_variants(list(forms))

    def _generate_generic_forms(self, phrase: str) -> List[str]:
        words = phrase.split()
        parsed = [self.morph.parse(w)[0] for w in words]

        all_forms = []
        cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']

        for case in cases:
            case_words = []
            for p in parsed:
                try:
                    form = p.inflect({case})
                    case_words.append(form.word if form else p.word)
                except Exception:
                    case_words.append(p.word)
            all_forms.append(" ".join(case_words))

        all_forms.append(phrase)
        return self._add_case_variants(list(set(all_forms)))

    def _expand_mapping(self, spans: List[EntitySpan]):
        for s in spans:
            self._remember(s.surface, s.etype)
            if s.etype == "PERSON":
                forms = self._generate_person_forms(s.surface)
            elif s.etype in ["ORG", "LOC"]:
                forms = self._generate_generic_forms(s.surface)
            else:
                forms = [s.surface]

            for form in forms:
                if form not in self.mapping:
                    self.mapping[form] = self.mapping[s.surface]

    def _apply_mapping(self, text: str) -> str:
        sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            tag = self.mapping[key]
            pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
            text = pattern.sub(tag, text)
        return text

    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        spans = self._extract_natasha_entities(text) + self._extract_lexicon_terms(text)
        self._expand_mapping(spans)

        if self.qwen:
            qwen_map = self.qwen.extract_domain_terms(text, self.mapping)
            for k, v in qwen_map.items():
                if k not in self.mapping:
                    self.mapping[k] = v

        anonymized = self._apply_mapping(text)
        self._save_mapping()
        return anonymized, self.mapping


if __name__ == "__main__":
    with open("new_meeting.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    qwen_client = QwenOllamaClient(QwenConfig())
    anonymizer = HybridAnonymizer(qwen=qwen_client, initial_dict_path="initial_dict.json")

    anonymized_text, mapping = anonymizer.anonymize(sample_text)

    with open("anonymized_text_{2}.txt", "w", encoding="utf-8") as f:
        f.write(anonymized_text)

    print("====DONE====")