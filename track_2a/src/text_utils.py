"""
Text normalisation and passage chunking shared by the booklet parser and the beginner task.
"""

import re
import unicodedata
from typing import List


def official_normalize(text: str) -> str:
    """Exact normalization from the pinned starter, without importing sklearn.

    Source: hackapertus-starter 5e6957729b77a126006049305ad1ef241211a5b4,
    evaluate.py. Parity is tested against that unmodified vendored implementation.
    """
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(\w) ?-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# "Personenfreizü -\ngigkeit" / "Zuwande-\nrung" -> "Personenfreizügigkeit" / "Zuwanderung"
_HYPHEN_BREAK = re.compile(r"(\w)\s?-\s*\n\s*([a-zäöüàâçéèêëîïôûùœ])")
_SENTENCE_END = re.compile(r"(?<=[.!?;:»])\s+(?=[A-ZÄÖÜÀÂÇÉÈÊËÎÏÔÛÙ«\"0-9])")


def dehyphenate(text: str) -> str:
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def normalise(text: str) -> str:
    return " ".join(dehyphenate(text).split())


def split_passages(text: str, max_chars: int = 600, min_chars: int = 80) -> List[str]:
    """
    Split text into passage-sized chunks: blank-line blocks first, then sentences.
    Chunks never exceed max_chars unless a single sentence does; tiny chunks are merged forward.
    """
    blocks = [normalise(b) for b in re.split(r"\n\s*\n", dehyphenate(text)) if b.strip()]
    pieces: List[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            pieces.append(block)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(block):
            if current and len(current) + len(sentence) + 1 > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    merged: List[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    return merged or ([normalise(text)] if text.strip() else [])


def _stems(text: str) -> set:
    """Accent-free 5-letter word stems plus numbers: a cheap bridge between DE/FR/IT cognates."""
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    stems = set()
    for w in re.findall(r"\w+", plain):
        if w.isdigit():
            stems.add(w)
        elif len(w) >= 5:
            stems.add(w[:5])
    return stems


def best_snippet(page_text: str, claim: str, max_chars: int = 360) -> str:
    """
    Pick the sentence(s) of a cited page that best match the claim, for display as evidence.
    Numbers count triple; repeated header fragments ("Texte soumis au vote Texte soumis au vote") are penalised.
    """
    sentences = [s for s in re.split(r"(?<=[.!?;:])\s+", normalise(page_text)) if len(s) >= 40]
    if not sentences:
        return normalise(page_text)[:max_chars]
    query = _stems(claim)

    def score(sentence: str) -> float:
        hits = query & _stems(sentence)
        value = sum(3.0 if h.isdigit() else 1.0 for h in hits)
        words = re.findall(r"\w+", sentence.lower())
        repetition = (len(words) - len(set(words))) / max(1, len(words))
        return value * (1.0 - repetition)

    best = max(range(len(sentences)), key=lambda i: score(sentences[i]))
    snippet = sentences[best]
    if len(snippet) < 160 and best + 1 < len(sentences):
        snippet = f"{snippet} {sentences[best + 1]}"
    snippet = re.sub(r"^\d{1,3}\s+", "", snippet)  # leading page number from the PDF text layer
    return snippet if len(snippet) <= max_chars else snippet[:max_chars].rsplit(" ", 1)[0] + " …"


def clip_to_query(page_text: str, query: str, max_chars: int, head_chars: int = 150) -> str:
    """
    Shorten an over-long page to the window of consecutive sentences that best matches the query
    (claim + vote title; the title is in the booklet language, so it also anchors cross-lingual claims).
    The page's first characters (page number, proposal title) are kept as a heading. A few dense
    legal-text pages reach 8-10k characters, against a median of ~1.4k.
    """
    if len(page_text) <= max_chars:
        return page_text
    text = normalise(page_text)
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    query_stems = _stems(query)
    scores = [sum(3.0 if h.isdigit() else 1.0 for h in (query_stems & _stems(s))) for s in sentences]
    budget = max_chars - head_chars
    best_score, best_start, best_end = -1.0, 0, 0
    for start in range(len(sentences)):
        total, length, end = 0.0, 0, start
        while end < len(sentences) and length + len(sentences[end]) + 1 <= budget:
            total += scores[end]
            length += len(sentences[end]) + 1
            end += 1
        if total > best_score:
            best_score, best_start, best_end = total, start, end
    window = " ".join(sentences[best_start:best_end]) or text[:budget]
    if best_start == 0:
        return window
    return f"{text[:head_chars].rsplit(' ', 1)[0]} … {window}"


_LANG_HINTS = {
    "de": {"der", "die", "das", "und", "nicht", "ist", "wird", "dass", "mit", "für", "den", "eine", "laut", "bundesrat"},
    "fr": {"le", "la", "les", "des", "et", "est", "que", "une", "pour", "dans", "du", "selon", "conseil", "fédéral"},
    "it": {"il", "lo", "gli", "della", "che", "è", "per", "una", "del", "non", "secondo", "consiglio", "federale", "di"},
}


def guess_language(text: str) -> str:
    """Cheap DE/FR/IT detection from function words (the OST input format has no language field)."""
    words = re.findall(r"[a-zàâçéèêëîïôûùüäöè]+", text.lower())
    scores = {lang: sum(w in hints for w in words) for lang, hints in _LANG_HINTS.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "de"
