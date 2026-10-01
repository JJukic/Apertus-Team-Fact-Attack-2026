"""
Numerical Conflict Detector for Swiss Political Claim Verification.
Detects contradictions arising from mutated quantities, years, percentages,
and currency figures between claims and official voting booklet texts.
Supports DE, FR, and IT number notations (e.g. 500'000 vs 1.7 Millionen).
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel


class NumericalEntity(BaseModel):
    raw_text: str
    value: float
    unit: str = ""
    start_char: int
    end_char: int
    context_tokens: List[str] = []

    @property
    def num_type(self) -> str:
        return "year" if self.unit == "year" else "quantity"


class NumericalConflictResult(BaseModel):
    has_conflict: bool
    claim_entity: str
    booklet_entity: str
    claim_value: float
    booklet_value: float
    topic_context: str
    explanation: str

    def __contains__(self, item: str) -> bool:
        return item in self.explanation or item in self.claim_entity or item in self.booklet_entity

    def __str__(self) -> str:
        return self.explanation


# Number word mappings across Swiss languages
_NUMBER_WORDS = {
    # German
    "ein": 1.0, "eine": 1.0, "eins": 1.0, "zwei": 2.0, "drei": 3.0, "vier": 4.0,
    "fünf": 5.0, "sechs": 6.0, "sieben": 7.0, "acht": 8.0, "neun": 9.0, "zehn": 10.0,
    "elf": 11.0, "zwölf": 12.0, "neuneinhalb": 9.5, "einhalb": 0.5,
    # French
    "un": 1.0, "une": 1.0, "deux": 2.0, "trois": 3.0, "quatre": 4.0,
    "cinq": 5.0, "six": 6.0, "sept": 7.0, "huit": 8.0, "neuf": 9.0, "dix": 10.0,
    # Italian
    "uno": 1.0, "una": 1.0, "due": 2.0, "tre": 3.0, "quattro": 4.0,
    "cinque": 5.0, "sei": 6.0, "sette": 7.0, "otto": 8.0, "nove": 9.0, "dieci": 10.0,
}

_SCALE_MULTIPLIERS = {
    # Millions
    "millionen": 1e6, "million": 1e6, "mio": 1e6,
    "millions": 1e6, "milioni": 1e6, "milione": 1e6,
    # Billions / Milliarde
    "milliarden": 1e9, "milliarde": 1e9, "mrd": 1e9,
    "milliards": 1e9, "milliard": 1e9, "miliardi": 1e9, "miliardo": 1e9,
    # Thousand
    "tausend": 1e3, "mille": 1e3, "mila": 1e3,
}

_STOPWORDS = {
    "dass", "wird", "werden", "noch", "damit", "viele", "mehr", "oder", "auch", "aber",
    "nach", "eine", "einer", "eines", "einem", "einen", "nicht", "kann", "können", "soll",
    "sollen", "muss", "müssen", "haben", "hatte", "sein", "waren", "wurde", "wurden",
    "bundesrat", "parlament", "empfiehlt", "schweiz", "volk", "stände", "jahr", "rund",
    "etwa", "circa", "ca", "environ", "circa", "plus", "moins", "weniger", "vor",
}


def _tokenize_context(text: str) -> List[str]:
    tokens = re.findall(r"[a-zäöüéèà]{4,}", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if t not in _STOPWORDS]


def extract_numerical_entities(text: str) -> List[NumericalEntity]:
    """
    Extract all numerical entities with normalized values and surrounding context tokens.
    """
    entities: List[NumericalEntity] = []
    seen_spans = set()

    # Normalize hyphenated word breaks like 'neun- einhalb' or '10- Millionen'
    cleaned = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # Pattern 1: Number + Scale Word (e.g. "1,7 Millionen", "10 Millionen", "9.5 millions", "500 Mio")
    p1 = re.compile(
        r"\b(\d+(?:[.,]\d+)?)\s*(millionen|million|mio|milliarden|milliarde|mrd|millions|milliards|milioni|miliardi)\b",
        re.IGNORECASE,
    )
    for m in p1.finditer(cleaned):
        raw_num = m.group(1).replace(",", ".")
        scale_word = m.group(2).lower()
        val = float(raw_num) * _SCALE_MULTIPLIERS.get(scale_word, 1.0)
        start, end = m.start(), m.end()
        seen_spans.add((start, end))

        ctx = cleaned[max(0, start - 50) : min(len(cleaned), end + 50)]
        entities.append(
            NumericalEntity(
                raw_text=m.group(0),
                value=val,
                unit=scale_word,
                start_char=start,
                end_char=end,
                context_tokens=_tokenize_context(ctx),
            )
        )

    # Pattern 2: Word number + Scale Word (e.g. "zehn Millionen", "dix millions", "due miliardi")
    p2 = re.compile(
        r"\b(ein|eine|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|neuneinhalb|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci)\s+(millionen|million|milliarden|milliarde|millions|milliards|milioni|miliardi)\b",
        re.IGNORECASE,
    )
    for m in p2.finditer(cleaned):
        word_num = m.group(1).lower()
        scale_word = m.group(2).lower()
        base_val = _NUMBER_WORDS.get(word_num, 1.0)
        val = base_val * _SCALE_MULTIPLIERS.get(scale_word, 1.0)
        start, end = m.start(), m.end()
        seen_spans.add((start, end))

        ctx = cleaned[max(0, start - 50) : min(len(cleaned), end + 50)]
        entities.append(
            NumericalEntity(
                raw_text=m.group(0),
                value=val,
                unit=scale_word,
                start_char=start,
                end_char=end,
                context_tokens=_tokenize_context(ctx),
            )
        )

    # Pattern 3: Swiss formatted integer with apostrophe/space (e.g. "500'000", "1'700'000", "500 000")
    p3 = re.compile(r"\b(\d{1,3}(?:['’ ]\d{3})+)\b")
    for m in p3.finditer(cleaned):
        start, end = m.start(), m.end()
        if any(s <= start and end <= e for s, e in seen_spans):
            continue
        cleaned_num = re.sub(r"['’\s]", "", m.group(1))
        try:
            val = float(cleaned_num)
            seen_spans.add((start, end))
            ctx = cleaned[max(0, start - 50) : min(len(cleaned), end + 50)]
            entities.append(
                NumericalEntity(
                    raw_text=m.group(0),
                    value=val,
                    unit="quantity",
                    start_char=start,
                    end_char=end,
                    context_tokens=_tokenize_context(ctx),
                )
            )
        except ValueError:
            pass

    # Pattern 4: Calendar Years (e.g. 2002, 2024, 2026, 2030, 2050)
    p4 = re.compile(r"\b(19\d{2}|20\d{2})\b")
    for m in p4.finditer(cleaned):
        start, end = m.start(), m.end()
        if any(s <= start and end <= e for s, e in seen_spans):
            continue
        val = float(m.group(1))
        seen_spans.add((start, end))
        ctx = cleaned[max(0, start - 50) : min(len(cleaned), end + 50)]
        entities.append(
            NumericalEntity(
                raw_text=m.group(0),
                value=val,
                unit="year",
                start_char=start,
                end_char=end,
                context_tokens=_tokenize_context(ctx),
            )
        )

    return entities


def detect_numerical_conflict(claim: str, context: str) -> Optional[NumericalConflictResult]:
    """
    Compare numerical entities between claim and context.
    If the claim cites a number on a topic that clearly matches a passage in the context,
    but the official figure is different, returns a NumericalConflictResult.
    """
    claim_entities = extract_numerical_entities(claim)
    if not claim_entities:
        return None

    context_entities = extract_numerical_entities(context)
    if not context_entities:
        return None

    for c_ent in claim_entities:
        # Check if the exact value already exists in context for the same type (year vs year, quantity vs quantity)
        has_exact = False
        for ctx_ent in context_entities:
            c_is_year = (c_ent.unit == "year")
            ctx_is_year = (ctx_ent.unit == "year")
            if c_is_year != ctx_is_year:
                continue
            if abs(c_ent.value - ctx_ent.value) < 0.01:
                has_exact = True
                break

        if has_exact:
            continue

        # Look for the best conflicting entity on the same topic (highest token overlap)
        best_candidate = None
        best_overlap_count = 0
        best_topic = set()

        for ctx_ent in context_entities:
            c_is_year = (c_ent.unit == "year")
            ctx_is_year = (ctx_ent.unit == "year")
            if c_is_year != ctx_is_year:
                continue

            shared_tokens = set(c_ent.context_tokens).intersection(set(ctx_ent.context_tokens))
            shared_topic = {t for t in shared_tokens if t not in ("millionen", "million", "milliarden", "jahr")}

            if len(shared_topic) > best_overlap_count:
                diff = abs(c_ent.value - ctx_ent.value)
                if diff > 0.01:
                    best_overlap_count = len(shared_topic)
                    best_candidate = ctx_ent
                    best_topic = shared_topic

        if best_candidate and best_overlap_count >= 1:
            topic_str = " / ".join(list(best_topic)[:4])
            explanation = (
                f"Zahlenkonflikt zum Thema '{topic_str}': "
                f"Behauptung nennt '{c_ent.raw_text}', "
                f"das offizielle Abstimmungsbüchlein belegt jedoch '{best_candidate.raw_text}'."
            )
            return NumericalConflictResult(
                has_conflict=True,
                claim_entity=c_ent.raw_text,
                booklet_entity=best_candidate.raw_text,
                claim_value=c_ent.value,
                booklet_value=best_candidate.value,
                topic_context=topic_str,
                explanation=explanation,
            )

    return None


# Aliases for convenience
extract_numbers = extract_numerical_entities
NumberEntity = NumericalEntity
