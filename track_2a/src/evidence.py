"""Source-only evidence extraction from source-attributed prediction pages.

The policies preserve NLI labels and never consume gold references. All quotes
are complete source pages or original PDF text blocks in the source language.
"""

import copy
from pathlib import Path
import re
import threading
import unicodedata

from src.text_utils import official_normalize as normalize


class SourcePages:
    def __init__(self, directory, lazy=False):
        self.directory = Path(directory)
        self.lazy = lazy
        self.cache = {}
        self._lock = threading.Lock()

    def get(self, case, page):
        if type(page) is not int or page < 1:
            return None
        filename = (self.directory / case['booklet']['path']).resolve()
        stat = filename.stat()
        identity = (str(filename), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self._lock:
            if self.lazy:
                key = (identity, page)
                if key not in self.cache:
                    self.cache[key] = self._extract_page(filename, page)
                return self.cache[key]
            if identity not in self.cache:
                self.cache[identity] = self._extract(filename)
            return self.cache[identity].get(page)

    @staticmethod
    def _extract_page(filename, number):
        import pymupdf
        with pymupdf.open(filename) as document:
            if number > len(document):
                return None
            return source_page(document[number - 1])

    @staticmethod
    def _extract(filename):
        return extract_source_pages(filename)


def source_page(page):
    """Text and text blocks of one PyMuPDF page (the form the gold passages follow)."""
    return {'text': page.get_text(sort=True),
            'blocks': [b[4] for b in page.get_text('blocks', sort=True) if b[6] == 0]}


def extract_source_pages(filename, numbers=None):
    """{page number: source page} for all pages (or the given 1-based numbers) of a PDF."""
    import pymupdf
    with pymupdf.open(filename) as document:
        wanted = range(1, len(document) + 1) if numbers is None else [n for n in numbers if 1 <= n <= len(document)]
        return {n: source_page(document[n - 1]) for n in wanted}


class ParsedSourcePages:
    """Source pages stored with the parsed booklet (parse cache): no PDF access, no lock, while cases are answered."""

    def __init__(self, pages):
        self.pages = pages

    def get(self, case, page):
        if type(page) is not int or page < 1:
            return None
        return self.pages.get(str(page)) or self.pages.get(page)


def terms(text):
    text = unicodedata.normalize('NFKD', (text or '').lower()).encode('ascii', 'ignore').decode()
    return {word if word.isdigit() else word[:5] for word in re.findall(r'\w+', text)
            if word.isdigit() or len(word) >= 5}


def construct_evidence(prediction, case, pages, policy):
    """Use prediction pages, including any upstream source-grounded fallback.

    Construction never adds an uncited page or reads an expected label/reference.
    """
    out = copy.deepcopy(prediction)
    if prediction['label'] == 1:
        out['evidence'] = []
        return out
    if 'booklet' not in case or prediction['label'] not in (0, 2):
        return out
    cited = []
    for item in prediction.get('evidence', []):
        if not isinstance(item, dict):
            continue
        number = item.get('page')
        page = pages.get(case, number)
        if page is not None and all(number != p for p, _ in cited):
            cited.append((number, page))
    full, blocks = [], []
    query = terms(case['claim']['text'])
    vote = terms(case.get('vote', ''))
    for priority, (number, page) in enumerate(cited):
        if page['text'].strip() and len(normalize(page['text'])) <= 5000:
            full.append({'page': number, 'text': page['text']})
        for order, block in enumerate(page['blocks']):
            # Preserve entire source blocks and their conditions/speaker context.
            # Avoid generic one-word quotations that can game partial matching.
            if not 80 <= len(normalize(block)) <= 5000:
                continue
            tokens = terms(block)
            score = sum(3 if word.isdigit() else 1 for word in query & tokens)
            score += 0.5 * len(vote & tokens)
            blocks.append((score, -priority, -order, {'page': number, 'text': block}))
    if policy == 'raw_pages':
        candidates = full
    elif policy == 'raw_pages_and_blocks':
        ranked = [b[3] for b in sorted(blocks, key=lambda b: b[:3], reverse=True)]
        candidates = full[:3] + ranked[:2] + full[3:] + ranked[2:]
    else:
        raise ValueError(f'Unknown evidence policy: {policy}')
    evidence, seen = [], set()
    for item in candidates:
        key = (item['page'], normalize(item['text']))
        if key not in seen:
            evidence.append(item)
            seen.add(key)
        if len(evidence) == 5:
            break
    out['evidence'] = evidence
    return out
