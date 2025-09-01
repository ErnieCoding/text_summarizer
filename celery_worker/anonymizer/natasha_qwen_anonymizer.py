from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import re
import requests
from tokenCounter import count_tokens

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
import os


# =========Константы=========

TAG_ORDER = [
    "PERSON", "ORG", "LOC", "DATE", "MONEY", "PERCENT", "NUMBER",
    "TECH", "PRODUCT", "PLATFORM"
]

DEFAULT_TECH_LEXICON = {
    "TECH": [
        "AI", "ML", "NLP", "LLM", "API", "SDK", "SaaS", "ERP", "CRM",
        "Python", "Java", "Go", "Rust", "PostgreSQL", "Kafka", "Kubernetes",
        "Qwen", "Qwen2", "Qwen3", "Llama", "Whisper", "Ollama"
    ],
    "PRODUCT": [
        "Salesforce", "SAP", "1C", "Bitrix24", "GitLab", "Jira", "Confluence"
    ],
    "PLATFORM": [
        "AWS", "GCP", "Azure", "Yandex Cloud"
    ]
}


@dataclass
class QwenConfig:
    host: str = "http://localhost:11434"
    endpoint: str = "/api/generate"
    model: str = "qwen3:30b"
    temperature: float = 0.4
    max_tokens: int = 3000


# =========Клиент Ollama=========

class QwenOllamaClient:
    def __init__(self, cfg: QwenConfig):
        self.cfg = cfg

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.host + self.cfg.endpoint
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def extract_domain_terms(self, text: str, already_found: Dict[str, str]) -> Dict[str, str]:
        """
        Просим Qwen вернуть JSON-словарь замен только для доменных сущностей:
        TECH / PRODUCT / PLATFORM. Все остальные категории игнорировать.
        """
        system = (
            "Ты помогаешь с анонимизацией протоколов. "
            "Выдели в тексте только доменные/нестандартные сущности: названия технологий, продуктов, платформ, кодовые имена проектов. "
            "Не включай имена людей, организации, локации, даты, деньги, числа — они уже покрыты. "
            "Верни только JSON с полем 'mappings', где ключ — исходная подстрока, значение — тег вида "
            "[TECH_xx], [PRODUCT_xx], [PLATFORM_xx]."
        )

        already = json.dumps(already_found, ensure_ascii=False)

        user = f"""Текст:
<<<
{text}
>>>

Уже найденные замены (не трогай):
{already}

Требования:
- Верни только JSON вида {{"mappings": {{ "Qwen": "[TECH_01]" }}}} без лишнего текста.
"""

        prompt = f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n"

        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "num_ctx": min(count_tokens(prompt) + 3000, 32768),
            }
        }
        data = self._post(payload)

        raw = data.get("response", "").strip()

        try:
            parsed = json.loads(raw)
            mappings = parsed.get("mappings", {})

            valid = {}
            for k, v in mappings.items():
                if isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"\[(TECH|PRODUCT|PLATFORM)_[0-9]{2}\]", v):
                    if k not in already_found:
                        valid[k] = v
            return valid
        except Exception as e:
            print(f"[ERROR] Модель не вернула валидный формат: {e}")
            return {}


# =========Анонимизатор=========

@dataclass
class EntitySpan:
    start: int
    end: int
    surface: str
    etype: str


class HybridAnonymizer:
    def __init__(
        self,
        qwen: Optional['QwenOllamaClient'] = None,
        tech_lexicon: Optional[Dict[str, List[str]]] = None
    ):
        # Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner = NewsNERTagger(self.emb)

        # ВАЖНО: Dates и Money требуют morph_vocab
        self.dates = DatesExtractor(self.morph_vocab)
        self.money = MoneyExtractor(self.morph_vocab)

        # Морфология
        self.morph = pymorphy2.MorphAnalyzer()

        # LLM клиент
        self.qwen = qwen

        # Лексикон доменных слов
        self.tech_lexicon = tech_lexicon or DEFAULT_TECH_LEXICON

        # Счетчики для тегов
        self.counters = defaultdict(int)

        # Глобальная карта замен (surface -> tag)
        self.mapping: Dict[str, str] = {}

    def _new_tag(self, etype: str) -> str:
        self.counters[etype] += 1
        return f"[{etype}_{self.counters[etype]:02d}]"

    def _remember(self, surface: str, etype: str) -> str:
        if surface not in self.mapping:
            self.mapping[surface] = self._new_tag(etype)
        return self.mapping[surface]

    def _extract_natasha_entities(self, text: str) -> List[EntitySpan]:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner)

        spans: List[EntitySpan] = []

        # PERSON / ORG / LOC
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            if span.type == "PER":
                spans.append(EntitySpan(span.start, span.stop, span.text, "PERSON"))
            elif span.type == "ORG":
                spans.append(EntitySpan(span.start, span.stop, span.text, "ORG"))
            elif span.type == "LOC":
                spans.append(EntitySpan(span.start, span.stop, span.text, "LOC"))

        # DATE
        for m in self.dates(text):
            surface = text[m.start:m.stop]
            spans.append(EntitySpan(m.start, m.stop, surface, "DATE"))

        # MONEY
        for m in self.money(text):
            surface = text[m.start:m.stop]
            spans.append(EntitySpan(m.start, m.stop, surface, "MONEY"))

        # PERCENT
        percent_regex = re.compile(r"(?<!\S)(\d+[.,]?\d*)\s*%")
        for mm in percent_regex.finditer(text):
            spans.append(EntitySpan(mm.start(1), mm.end(0), mm.group(0), "PERCENT"))

        # NUMBER
        for mm in re.finditer(r'\b\d+[.,]?\d*\b', text):
            val = mm.group(0)
            if not re.fullmatch(r'\d+[.,]?\d*%', val):
                spans.append(EntitySpan(mm.start(), mm.end(), val, "NUMBER"))

        return spans


    def _extract_lexicon_terms(self, text: str) -> List[EntitySpan]:
        spans: List[EntitySpan] = []
        for etype, words in self.tech_lexicon.items():
            for w in words:
                pattern = re.compile(rf"(?<!\w){re.escape(w)}(?!\w)", re.IGNORECASE)
                for m in pattern.finditer(text):
                    spans.append(EntitySpan(m.start(), m.end(), m.group(0), etype))
        return spans

    @staticmethod
    def _dedupe_and_sort_spans(spans: List[EntitySpan]) -> List[EntitySpan]:
        keyset = set()
        out = []
        for s in spans:
            k = (s.start, s.end, s.surface, s.etype)
            if k not in keyset:
                keyset.add(k)
                out.append(s)
        out.sort(key=lambda x: (x.start, -x.end))
        result = []
        last_end = -1
        for s in out:
            if s.start >= last_end:
                result.append(s)
                last_end = s.end
        result.sort(key=lambda x: x.start, reverse=True)
        return result

    def _build_mapping_from_spans(self, spans: List[EntitySpan]) -> None:
        for s in spans:
            self._remember(s.surface, s.etype)

    @staticmethod
    def _apply_spans(text: str, spans: List[EntitySpan], mapping: Dict[str, str]) -> str:
        out = text
        for s in spans:
            tag = mapping.get(s.surface)
            if not tag:
                continue
            out = out[:s.start] + tag + out[s.end:]
        return out

    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        natasha_spans = self._extract_natasha_entities(text)
        lex_spans = self._extract_lexicon_terms(text)

        tmp_spans = self._dedupe_and_sort_spans(natasha_spans + lex_spans)
        self._build_mapping_from_spans(tmp_spans)
        already_found = dict(self.mapping)

        llm_mapping: Dict[str, str] = {}
        if self.qwen is not None:
            try:
                llm_mapping = self.qwen.extract_domain_terms(text, already_found=already_found)
            except Exception:
                llm_mapping = {}

        llm_spans: List[EntitySpan] = []
        for surface, tag in llm_mapping.items():
            m = re.match(r"\[(TECH|PRODUCT|PLATFORM)_[0-9]{2}\]", tag)
            if not m:
                continue
            etype = m.group(1)
            for mm in re.finditer(re.escape(surface), text):
                llm_spans.append(EntitySpan(mm.start(), mm.end(), mm.group(0), etype))
                self.mapping[mm.group(0)] = tag

        all_spans = self._dedupe_and_sort_spans(natasha_spans + lex_spans + llm_spans)
        self._build_mapping_from_spans(all_spans)
        anonymized = self._apply_spans(text, all_spans, self.mapping)

        return anonymized, dict(self.mapping)


if __name__ == "__main__":
    with open("new_meeting.txt", "r", encoding="utf-8", errors="replace") as file:
        sample_text = file.read()

    qwen_client = QwenOllamaClient(QwenConfig())
    anonymizer = HybridAnonymizer(qwen=qwen_client)

    anonymized_text, mapping = anonymizer.anonymize(sample_text)

    with open("anonymized_txt_{1}.txt", "w+", encoding="utf-8") as file:
        file.write(anonymized_text)
    
    with open("mappings_{1}.txt", "w+", encoding="utf-8") as file:
        for k, v in mapping.items():
            file.write(f"{k!r} -> {v}\n")

    print("====DONE====")
