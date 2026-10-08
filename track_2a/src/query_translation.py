"""Guarded, source-language queries; the NLI hypothesis stays unchanged.

The cache contains only translations of supplied claim text, never labels or
reference passages. Remote acquisition must be measured by the request recorder.
Negation presence is a conservative heuristic, not a semantic equivalence proof.
"""

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import re
import threading
import time
import unicodedata

from src.apertus_client import ApertusClient


def numbers(text):
    text = unicodedata.normalize('NFKC', text)
    # Grouped thousands, apostrophes and decimal commas are normalized without
    # accepting changed values. Ambiguous formats fall back to the original query.
    text = re.sub(r"(?<=\d)[ '\u2019](?=\d{3}(?:\D|$))", '', text)
    values = re.findall(r'\d+(?:[.,]\d+)*', text)
    return Counter(value.replace(',', '.') for value in values)


def has_negation(text, language):
    pattern = {
        'de': r'\b(nicht|kein\w*|nie|ohne|verbot\w*|verbiet\w*)\b',
        'fr': r'\b(pas|jamais|aucun\w*|sans|interdi\w*)\b|\bn[’\']',
        'it': r'\b(non|nessun\w*|senza|viet\w*)\b',
    }.get(language, r'\b(not|no|without)\b')
    return bool(re.search(pattern, text, re.I))


def guard(original, translated, source_language, target_language):
    if not isinstance(translated, str) or not translated.strip() or '<|inner_' in translated:
        return 'missing_or_unclosed_translation'
    if numbers(original) != numbers(translated):
        return 'numeric_values_changed'
    if has_negation(original, source_language) != has_negation(translated, target_language):
        return 'negation_presence_changed'
    return None


class QueryTranslator:
    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.cache, self.locks = {}, {}
        self._lock = threading.Lock()
        self.prompt_hash = hashlib.sha256(inspect.getsource(ApertusClient.translate).encode()).hexdigest()

    def get(self, claim, claim_language, booklet_language):
        started = time.perf_counter()
        identity = {'claim': claim, 'claim_language': claim_language, 'booklet_language': booklet_language,
                    'model': self.client.model_name, 'translation_code_sha256': self.prompt_hash}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            path = self.directory / (key + '.json')
            record = self.cache.get(key)
            if record is None:
                try:
                    stored = json.loads(path.read_text())
                    if stored.get('identity') == identity:
                        record = stored
                except (OSError, ValueError):
                    pass
            if record is not None:
                self.cache[key] = record
                return (record['translation'] if record['rejection'] is None else None, 0, 0,
                        {'cache_key': key, 'cache_hit': True, 'rejection': record['rejection'],
                         'elapsed_ms': (time.perf_counter() - started) * 1000})
            translation, input_tokens, output_tokens, remote_ms = self.client.translate(claim, booklet_language)
            rejection = guard(claim, translation, claim_language, booklet_language)
            record = {'identity': identity, 'translation': translation, 'rejection': rejection,
                      'acquisition_usage_reported': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}
            # Do not cache a transport failure as a permanent missing translation.
            if translation is not None:
                self.cache[key] = record
                try:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(record, ensure_ascii=False))
                    temporary.replace(path)
                except OSError:
                    pass
            return (translation if rejection is None else None, input_tokens, output_tokens,
                    {'cache_key': key, 'cache_hit': False, 'rejection': rejection, 'remote_ms': remote_ms,
                     'elapsed_ms': (time.perf_counter() - started) * 1000})
