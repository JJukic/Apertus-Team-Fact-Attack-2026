"""Offline scoring integration; never read gold data during inference.

Use the pinned starter implementation directly, including normalization, first
five positions, post-normalization length limits and fuzzy matching threshold.
"""

from vendor.hackapertus_starter.evaluate import (
    INVALID, LABELS, MAX_QUOTES, MAX_QUOTE_CHARS, THRESHOLDS,
    label_of, macro_f1, normalize, quote_matches, quotes_of,
)


def evidence_hit5(prediction: dict, reference: str) -> bool:
    return any(quote_matches(quote, reference, 90, MAX_QUOTE_CHARS)
               for quote in quotes_of(prediction))
