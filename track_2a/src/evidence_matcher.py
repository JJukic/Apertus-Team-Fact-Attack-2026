"""Conservative quote matching with offsets into the original PDF text.

No fuzzy similarity, translation, word deletion or cross-page stitching.
PDF word breaks are removed only between letters, never between numbers.
"""

import re

_PAGE = re.compile(r"\[(?:page|seite)\s+(\d+)\]", re.I)
_PREFIX = re.compile(r"^\s*(?:\[(?:Section|Proposal|Speaker)\s+[^\]]*\]\s*)+", re.I)
_BREAK = re.compile(
    r"(?<=[^\W\d_])(?:[-\u00ad]\s+|[ \t]+-\s*\n\s*)(?=[^\W\d_])",
    re.UNICODE,
)


def _normalize(text):
    skipped = set()
    for match in _BREAK.finditer(text):
        skipped.update(range(match.start(), match.end()))
    chars, offsets = [], []
    for i, char in enumerate(text):
        if i in skipped or char == "\u00ad":
            continue
        # Some model outputs escape a typographic dash as if it were Markdown.
        if char == "\\" and i + 1 < len(text) and text[i + 1] in "–—":
            continue
        value = " " if char.isspace() else char.lower()
        if char in "–—":
            value = "–"
        if value == " " and chars and chars[-1] == " ":
            offsets[-1] = (offsets[-1][0], i + 1)
            continue
        for part in value:
            chars.append(part)
            offsets.append((i, i + 1))
    return "".join(chars), offsets


def match_quote(evidence, sources):
    """Return (source, original span), or None; source IDs always come from PDF.

    A model page tag is only a tie breaker among textually verified matches.
    """
    tag = _PAGE.search(evidence)
    clean = _PAGE.sub("", evidence, count=1) if tag and not evidence[:tag.start()].strip(' \"«»') else evidence
    clean = _PREFIX.sub("", clean).strip(' \t\r\n\"«»“”')
    needle, _ = _normalize(clean)
    needle = needle.strip()
    if len(needle) < 15:
        return None
    matches = []
    for source in sources:
        original = source["text"]
        haystack, offsets = _normalize(original)
        start = haystack.find(needle)
        while start >= 0:
            end = start + len(needle)
            left_ok = not (needle[0].isalnum() and start and haystack[start - 1].isalnum())
            right_ok = not (needle[-1].isalnum() and end < len(haystack) and haystack[end].isalnum())
            if left_ok and right_ok:
                matches.append((source, original[offsets[start][0]:offsets[end - 1][1]]))
                break
            start = haystack.find(needle, start + 1)
    if tag:
        preferred = [m for m in matches if m[0]["page_number"] == int(tag.group(1))]
        if preferred:
            return preferred[0]
    return matches[0] if matches else None
