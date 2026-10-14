"""
Contents-guided part selection for task A (PART_CONTEXT): send Apertus only the booklet part a claim refers to.

- Claims name their source in the opening ("Laut der Zusammenfassung …", "Le comité estime …", "Se il voto viene
  accettato …"); `claim_part` maps the earliest such cue to one of five parts.
- Every booklet since 2020 opens with a table of contents that lists, per vote, the pages of its parts
  ("In Kürze 4 – 5 | Im Detail 14 | Argumente 20 | Abstimmungstext 24"); `booklet_parts` reads it and checks each
  start page against the page's own heading. The arguments are split into the committee's and the Federal Council's
  pages with the parser's speaker labels.
- `part_pages` returns the pages of the claim's part of the case's vote, or None (then the regular retrieval runs).
"""
import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

SUMMARY, DETAIL, ARGUMENTS, LEGAL = "summary", "detail", "arguments", "legal_text"
COMMITTEE, FEDERAL_COUNCIL = "committee", "federal_council"

# Table-of-contents entries (DE / FR / IT), each followed by a page number or a range "4 – 5"
_TOC_ENTRIES = [
    (SUMMARY, re.compile(r"^(?:das\s+wichtigste\s+)?in\s+k[uü]rze$|^(?:l.essentiel\s+)?en\s+bref$|^in\s+breve$", re.I)),
    (DETAIL, re.compile(r"^(?:vorlage\s+)?im\s+detail$|^en\s+d[eé]tail$|^in\s+dettaglio$", re.I)),
    (ARGUMENTS, re.compile(r"^argumente$|^arguments$|^(?:gli\s+)?argomenti$", re.I)),
    (LEGAL, re.compile(r"^abstimmungstext$|^textes?\s+soumis\s+au\s+vote$|^(?:il\s+)?test[oi]\s+in\s+votazione$", re.I)),
]
_PAGE_REF = re.compile(r"^(\d{1,3})(?:\s*[–-]\s*(\d{1,3}))?$")
# Lines of the contents page that are not part of a vote title
_TOC_NOISE = re.compile(
    r"^(?:\d{1,3}|inhalt|inhaltsverzeichnis|sommaire|sommario|indice"
    r"|(?:erste|zweite|dritte|vierte|f[uü]nfte|sechste)\s+vorlage"
    r"|(?:premier|deuxi[eè]me|troisi[eè]me|quatri[eè]me|cinqui[eè]me|sixi[eè]me)\s+objet"
    r"|(?:primo|secondo|terzo|quarto|quinto|sesto)\s+oggetto)$", re.I)

# Claim openings -> part. The earliest cue in the claim wins ("Le Conseil fédéral recommande d'accepter …" is a
# Federal Council claim, "Si le vote est accepté, le Conseil fédéral …" a consequence of a yes).
_CLAIM_CUES = [
    (SUMMARY, re.compile(r"zusammenfassung|r[eé]sum[eé]|riassunto|riepilogo|in\s+k[uü]rze|en\s+bref|in\s+breve"
                         r"|kurz[uü]bersicht|[uü]berblick|aper[cç]u|synth[eè]se|sintesi|panoramica", re.I)),
    (COMMITTEE, re.compile(r"komitee|initiant|comit[eé]|comitato|promotori"
                           r"|bef[uü]rworter|urheber|partisans?\b|auteurs?\s+de\s+l|sostenitor[ie]|autori\s+dell", re.I)),
    (FEDERAL_COUNCIL, re.compile(r"bundesrat|conseil\s+f[eé]d[eé]ral|consiglio\s+federale|parlament"
                                 r"|landesregierung|gouvernement|governo", re.I)),
    (LEGAL, re.compile(r"abstimmungstext|gesetzestext|verfassungstext|texte\s+(?:soumis|de\s+(?:vote|la\s+votation))"
                       r"|test[oi]\s+(?:di|in|della|sottopost[oi])\s*(?:voto|votazione|al\s+voto)?"
                       r"|wortlaut|verfassungsartikel|gesetzesartikel|libell[eé]|article\s+constitutionnel"
                       r"|texte\s+(?:de\s+loi|l[eé]gal)|articolo\s+costituzionale|testo\s+del\s+(?:progetto|decreto)", re.I)),
    (DETAIL, re.compile(r"angenommen|annahme|si\s+(?:le|la|cette)\s+(?:vote|votation|proposition|motion|objet)"
                        r"|en\s+cas\s+d.(?:acceptation|adoption)|l.adoption|se\s+(?:il|la)\s+(?:voto|votazione|proposta)"
                        r"|in\s+caso\s+di\s+(?:s[iì]|accettazione|approvazione)"
                        r"|bei\s+einem\s+ja\b|en\s+cas\s+de\s+oui|con\s+un\s+s[iì]\b|in\s+caso\s+di\s+un\s+s[iì]", re.I)),
]
_CUE_WINDOW = 80  # the source is named at the start of the claim
# A cue counts only if it starts this early (the dataset's and the holdout's start by character 20 at the latest): a
# source named later, or an actor mentioned in the claim's content ("L'iniziativa è stata respinta dal Consiglio
# federale …"), leaves the claim to the regular retrieval instead of sending it to the wrong part
_CUE_START = 30


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def claim_part(claim: str) -> Optional[str]:
    """The part a claim names in its opening (summary, committee, federal_council, legal_text, detail), if any."""
    head = claim[:_CUE_WINDOW]
    found = [(m.start(), part) for part, pattern in _CLAIM_CUES for m in [pattern.search(head)]
             if m and m.start() <= _CUE_START]
    return min(found)[1] if found else None


def _toc_lines(text: str) -> List[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def parse_toc(page_text: str) -> List[Dict[str, Any]]:
    """Votes listed on a contents page: [{'title', 'parts': {part: (first, last or None)}}] in listing order."""
    lines = _toc_lines(page_text)
    votes: List[Dict[str, Any]] = []
    title: List[str] = []
    current: Optional[Dict[str, Any]] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        part = next((p for p, pattern in _TOC_ENTRIES if pattern.match(line)), None)
        # the page number is on the same line ("In Kürze 4 – 5") or on the next one
        ref = None
        if part is None:
            m = re.match(r"^(.*?)\s+(\d{1,3}(?:\s*[–-]\s*\d{1,3})?)$", line)
            if m:
                part = next((p for p, pattern in _TOC_ENTRIES if pattern.match(m.group(1))), None)
                ref = _PAGE_REF.match(m.group(2)) if part else None
        elif i + 1 < len(lines):
            ref = _PAGE_REF.match(lines[i + 1])
            if ref:
                i += 1
        if part and ref:
            if part == SUMMARY or current is None or part in current["parts"]:
                current = {"title": " ".join(t for t in title if not _TOC_NOISE.match(t)), "parts": {}}
                votes.append(current)
                title = []
            first, last = int(ref.group(1)), int(ref.group(2)) if ref.group(2) else None
            current["parts"][part] = (first, last)
        elif not _PAGE_REF.match(line):
            title.append(line)
        i += 1
    # Real entries list the summary pages; the navigation bars on the summary pages repeat the other parts only
    return [v for v in votes if SUMMARY in v["parts"] and len(v["parts"]) >= 3 and v["title"]]


def _heading_zone(text: str) -> str:
    return _fold(" ".join(_toc_lines(text)[:6]))


_PART_HEADINGS = {
    SUMMARY: re.compile(r"in kurze|en bref|in breve"),
    DETAIL: re.compile(r"im detail|en detail|in dettaglio"),
    ARGUMENTS: re.compile(r"argument|argomenti|komitee|comite|comitato|bundesrat|conseil federal|consiglio federale"),
    LEGAL: re.compile(r"abstimmungstext|soumis au vote|in votazione|§"),
}


def booklet_parts(page_texts: Dict[int, str], speaker_of: Dict[int, Optional[str]]) -> Optional[List[Dict[str, Any]]]:
    """
    Page ranges of every vote's parts from the contents page, or None if no consistent contents page is found.
    `page_texts`: 1-based page -> text; `speaker_of`: page -> 'committee' / 'federal_council' / None (parser labels).
    """
    # a page-by-page reader (src/quick_booklet.py) knows its last page without reading every page
    n_pages = page_texts.last_page() if hasattr(page_texts, "last_page") else (max(page_texts) if page_texts else 0)
    # The contents may continue on a second page (five votes on 27 September 2020)
    votes, seen = [], set()
    for p in [p for p in range(1, 7) if p in page_texts]:
        for v in parse_toc(page_texts[p]):
            key = tuple(sorted(first for first, _ in v["parts"].values()))
            if key not in seen:  # the same contents can be printed twice (cover flap and inside page)
                seen.add(key)
                votes.append(v)
    if votes:
        # Printed page numbers normally equal PDF page numbers; accept a small offset if the headings say so
        best = None
        for offset in (0, 1, -1, 2, -2):
            checks = [_PART_HEADINGS[part].search(_heading_zone(page_texts.get(first + offset, "")))
                      for v in votes for part, (first, _) in v["parts"].items() if part in (DETAIL, ARGUMENTS, LEGAL)]
            score = sum(bool(c) for c in checks) / max(len(checks), 1)
            if best is None or score > best[0]:
                best = (score, offset)
            if score == 1.0:  # no later offset can score higher (ties keep the earlier one): skip reading their pages
                break
        if best[0] < 0.5:
            return None
        offset = best[1]
        starts = sorted({first + offset for v in votes for first, _ in v["parts"].values()})
        result = []
        for v in votes:
            ranges = {}
            for part, (first, last) in v["parts"].items():
                first += offset
                if last is not None:
                    last += offset
                else:
                    later = [s for s in starts if s > first]
                    last = (later[0] - 1) if later else min(n_pages, first + 5)
                ranges[part] = list(range(first, min(last, n_pages) + 1))
            if ARGUMENTS in ranges and not callable(speaker_of):
                ranges[COMMITTEE], ranges[FEDERAL_COUNCIL] = split_arguments(ranges[ARGUMENTS], speaker_of)
            # The contents abbreviate some titles; the first lines of the summary and detail pages repeat them in full
            names = [v["title"]] + [" ".join(_toc_lines(page_texts.get(ranges[p][0], ""))[:8])
                                    for p in (SUMMARY, DETAIL) if ranges.get(p)]
            result.append({"title": v["title"], "names": names, "parts": ranges,
                           "speakers": speaker_of if callable(speaker_of) else None})
        return result
    return None


def split_arguments(pages: List[int], speaker_of: Dict[int, Optional[str]]) -> Tuple[List[int], List[int]]:
    """(committee pages, Federal Council pages) of an arguments range, by the parser's speaker labels."""
    committee = [p for p in pages if speaker_of.get(p) == COMMITTEE]
    council = [p for p in pages if speaker_of.get(p) == FEDERAL_COUNCIL]
    if not committee or not council:  # labels missing: the committee's double page comes first
        half = (len(pages) + 1) // 2
        committee, council = committee or pages[:half], council or pages[half:]
    return committee, council


def _words(text: str) -> set:
    return {w for w in _fold(text).split() if len(w) >= 3 and not w.isdigit()}


def match_vote(votes: List[Dict[str, Any]], vote: Optional[str], min_score: float = 0.5,
               margin: float = 0.1) -> Optional[Dict[str, Any]]:
    """The listed vote whose names share most of the case's vote title words (contents titles wrap and abbreviate)."""
    if not votes or not vote:
        return None
    target = _words(vote)
    if not target:
        return None
    scored = sorted(((max(len(target & _words(n)) / len(target) for n in v.get("names", [v["title"]])), k)
                     for k, v in enumerate(votes)), reverse=True)
    best, k = scored[0]
    if best < min_score or (len(scored) > 1 and best - scored[1][0] < margin):
        return None
    return votes[k]


def part_pages(claim: str, vote: Optional[str], parts: Optional[List[Dict[str, Any]]]) -> Optional[Tuple[str, List[int]]]:
    """(part, pages) of the claim's part of the case's vote, or None when the claim, the vote or the part is unknown."""
    part = claim_part(claim)
    if part is None or not parts:
        return None
    v = match_vote(parts, vote)
    if v is None:
        return None
    if part in (COMMITTEE, FEDERAL_COUNCIL) and part not in v["parts"] and ARGUMENTS in v["parts"] and v.get("speakers"):
        # speaker labels given as a function (booklets read page by page): split this vote's arguments only
        args = v["parts"][ARGUMENTS]
        v["parts"][COMMITTEE], v["parts"][FEDERAL_COUNCIL] = split_arguments(args, v["speakers"](args))
    pages = v["parts"].get(part)
    return (part, pages) if pages else None
