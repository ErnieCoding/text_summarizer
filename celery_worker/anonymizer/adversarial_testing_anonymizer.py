from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import re
import os
import requests
import time

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

    def adversarial_guess(self, anonymized_text: str, attack_prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[Dict[str, str]]:
        """
        Отправляет атакующий промпт на модель, возвращает JSON mapping guesses.
        Ожидаем от модели JSON вида {"[PERSON_01]": "Иван Иванов", ...}
        """
        system = (
            "Ты — аналитик текста. Твоя задача — по анонимизированному тексту попытаться восстановить "
            "реальные сущности (имена, организации, локации, продукты). "
            "Возвращай строго JSON: ключ — тег из текста (например, \"[PERSON_01]\"), значение — восстановленная строка."
        )

        user = f"""Анонимизированный текст:
<<<
{anonymized_text}
>>>

Инструкция:
{attack_prompt}

Возвращай только JSON-объект без пояснений.
"""
        payload = {
            "model": self.cfg.model,
            "prompt": f"<s>[SYSTEM]{system}[/SYSTEM]\n[USER]{user}[/USER]",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            resp = self._post(payload)
            raw = resp.get("response", "").strip()
            # Иногда модель возвращает текст до/после JSON — попробуем извлечь JSON
            # Быстрая эвристика: найти первый "{" и последний "}" и парсить
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw[start:end+1]
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    # если парсинг JSON не удался — возвращаем None
                    return None
            return None
        except Exception:
            return None


# ========= EntitySpan =========
@dataclass
class EntitySpan:
    start: int
    end: int
    surface: str
    etype: str


# ========= Adversarial Tester =========
class AdversarialTester:
    def __init__(self, qwen_client: QwenOllamaClient):
        self.qwen = qwen_client
        # набор атак — разные промпты и параметры
        self.attacks = [
            {
                "name": "direct_guess",
                "prompt": "Попробуй по тексту угадать реальные значения для всех тегов. Если не уверен — поставь null.",
                "temperature": 0.4
            },
            {
                "name": "creative_guess",
                "prompt": "Используя контекст, выдвинь наиболее вероятные варианты для каждого тега. Если несколько вариантов — перечисли их через ;",
                "temperature": 0.8
            },
            {
                "name": "hinted_guess",
                "prompt": "Попробуй восстановить сущности. Особенно обращай внимание на контекст, даты и продукты.",
                "temperature": 0.6
            }
        ]

    def run_attacks(self, anonymized_text: str) -> Dict[str, Dict[str, Any]]:
        results = {}
        for attack in self.attacks:
            name = attack["name"]
            prompt = attack["prompt"]
            temp = attack["temperature"]
            guessed = self.qwen.adversarial_guess(anonymized_text, prompt, temperature=temp, max_tokens=1200)
            results[name] = {
                "guesses": guessed or {},
                "raw_prompt": prompt,
                "temperature": temp
            }
            # короткая пауза, чтобы не перегружать локальный сервер
            time.sleep(0.2)
        return results

    @staticmethod
    def evaluate_guesses(guesses: Dict[str, str], canonical_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Сравниваем guesses (tag->string) с canonical_map (tag->canonical_surface).
        Возвращаем метрики: matched_tags, total_tags, accuracy (пер-тегов), partial_matches (если частичное совпадение)
        """
        total = len(canonical_map)
        matched = 0
        partial = 0
        details = {}

        for tag, true_val in canonical_map.items():
            guess = guesses.get(tag)
            if not guess:
                details[tag] = {"guess": None, "true": true_val, "match": False}
                continue
            # нормализуем строки (простая нормализация)
            g = guess.strip().lower()
            t = true_val.strip().lower()
            if g == t:
                matched += 1
                details[tag] = {"guess": guess, "true": true_val, "match": True}
            else:
                # частичное совпадение: проверим, содержится ли true в guess или наоборот
                if t in g or g in t:
                    partial += 1
                    details[tag] = {"guess": guess, "true": true_val, "match": "partial"}
                else:
                    details[tag] = {"guess": guess, "true": true_val, "match": False}

        accuracy = matched / total if total else 0.0
        partial_rate = partial / total if total else 0.0
        return {
            "total_tags": total,
            "matched": matched,
            "partial": partial,
            "accuracy": accuracy,
            "partial_rate": partial_rate,
            "details": details
        }


# ========= Анонимизатор с adversarial testing =========
class HybridAnonymizer:
    def __init__(
        self,
        qwen: Optional[QwenOllamaClient] = None,
        tech_lexicon: Optional[Dict[str, List[str]]] = None,
        initial_dict_path: str = "initial_dict_{3}.json"
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

        # Для adversarial testing: tag -> canonical original surface (the representative)
        self.canonical_for_tag: Dict[str, str] = {}
        # также обратный индекс tag -> set(surfaces)
        self.tag_to_surfaces: Dict[str, set] = defaultdict(set)

        # adversarial tester
        self.adversary = AdversarialTester(qwen) if qwen else None

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
        """
        Запоминаем surface -> tag. Если тэг новый — регистрируем canonical_for_tag.
        """
        if surface not in self.mapping:
            tag = self._new_tag(etype)
            self.mapping[surface] = tag
            # помечаем canonical представителем для тега
            if tag not in self.canonical_for_tag:
                self.canonical_for_tag[tag] = surface
            self.tag_to_surfaces[tag].add(surface)

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
        """
        Запоминаем канонические сущности и добавляем падежные/регистр-формы в mapping.
        """
        for s in spans:
            # canonical registration happens here
            self._remember(s.surface, s.etype)

        # После регистрации canonical surfaces, добавим формы для каждого canonical
        for s in spans:
            tag = self.mapping[s.surface]  # canonical tag
            if s.etype == "PERSON":
                forms = self._generate_person_forms(s.surface)
            elif s.etype in ["ORG", "LOC"]:
                forms = self._generate_generic_forms(s.surface)
            else:
                forms = [s.surface]

            for form in forms:
                if form not in self.mapping:
                    self.mapping[form] = tag
                    self.tag_to_surfaces[tag].add(form)

    def _apply_mapping(self, text: str) -> str:
        sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            tag = self.mapping[key]
            pattern = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
            text = pattern.sub(tag, text)
        return text

    def _run_adversarial_tests(self, anonymized_text: str) -> Dict[str, Any]:
        """
        Запускаем adversarial attacks через AdversarialTester и оцениваем результаты.
        """
        if not self.adversary:
            return {"error": "No adversary configured (no Qwen client)"}

        # запускаем несколько атак
        attacks_results = self.adversary.run_attacks(anonymized_text)

        # строим canonical_map: tag -> canonical surface (representative)
        canonical_map = dict(self.canonical_for_tag)  # shallow copy

        # Оцениваем каждую атаку
        evals = {}
        for attack_name, attack_res in attacks_results.items():
            guesses = attack_res.get("guesses") or {}
            eval_report = AdversarialTester.evaluate_guesses(guesses, canonical_map)
            evals[attack_name] = {
                "attack_meta": {"temperature": attack_res.get("temperature"), "raw_prompt": attack_res.get("raw_prompt")},
                "guesses": guesses,
                "evaluation": eval_report
            }

        # агрегированный summary
        summary = {
            "num_tags": len(canonical_map),
            "attacks": {k: v["evaluation"] for k, v in evals.items()}
        }

        report = {
            "summary": summary,
            "attacks": evals,
            "canonical_map": canonical_map
        }

        # сохраняем локально
        try:
            with open("adversarial_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return report

    def anonymize(self, text: str, run_adversarial: bool = True) -> Tuple[str, Dict[str, str], Optional[Dict[str, Any]]]:
        spans = self._extract_natasha_entities(text) + self._extract_lexicon_terms(text)
        self._expand_mapping(spans)

        if self.qwen:
            qwen_map = self.qwen.extract_domain_terms(text, self.mapping)
            for k, v in qwen_map.items():
                if k not in self.mapping:
                    self.mapping[k] = v
                    # register canonical for domain term if needed
                    if v not in self.canonical_for_tag:
                        self.canonical_for_tag[v] = k
                    self.tag_to_surfaces[v].add(k)

        anonymized = self._apply_mapping(text)
        self._save_mapping()

        adversarial_report = None
        if run_adversarial and self.adversary:
            adversarial_report = self._run_adversarial_tests(anonymized)

        return anonymized, dict(self.mapping), adversarial_report


# ========= MAIN =========
if __name__ == "__main__":
    with open("new_meeting.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    qwen_client = QwenOllamaClient(QwenConfig())
    anonymizer = HybridAnonymizer(qwen=qwen_client, initial_dict_path="initial_dict_{3}.json")

    anonymized_text, mapping, adversarial_report = anonymizer.anonymize(sample_text, run_adversarial=True)

    with open("anonymized_text_{3}.txt", "w", encoding="utf-8") as f:
        f.write(anonymized_text)

    # вывод сводки из отчёта
    if adversarial_report:
        summary = adversarial_report.get("summary", {})
        print("==== ADVERSARIAL SUMMARY ====")
        print("Number of tags:", summary.get("num_tags"))
        for attack_name, eval_data in summary.get("attacks", {}).items():
            print(f"Attack: {attack_name}")
            print(f"  accuracy: {eval_data.get('accuracy'):.3f}, partial_rate: {eval_data.get('partial_rate'):.3f}")
    else:
        print("Adversarial testing not run (no Qwen client or run_adversarial=False).")

    print("====DONE====")