"""
Text normalisation and passage chunking shared by the booklet parser and the beginner task.
"""

import re
import unicodedata
from typing import List

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
