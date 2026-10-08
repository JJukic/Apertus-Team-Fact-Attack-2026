"""
Configuration and Environment Management for Hack Apertus Track 2A (OST).
"""

import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv

_RUNTIME_LLM_ENV = {name: os.environ[name] for name in
                    ("BASE_URL", "API_KEY", "LLM_BASE_URL", "LLM_API_KEY") if name in os.environ}

# Search for .env in track_2a directory or repo root
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

if (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env")
elif (REPO_ROOT / ".env").exists():
    load_dotenv(REPO_ROOT / ".env")
else:
    load_dotenv()

# Apertus / LLM Endpoint Config
LLM_NAME = os.getenv("LLM_NAME", "swiss-ai/Apertus-v1.5-70B-thinking")
def _endpoint_setting(official, legacy):
    # Any runtime setting outranks a .env value, including across alias names.
    for name in (official, legacy):
        if name in _RUNTIME_LLM_ENV:
            return _RUNTIME_LLM_ENV[name]
    return os.getenv(official) or os.getenv(legacy) or ""


LLM_BASE_URL = _endpoint_setting("BASE_URL", "LLM_BASE_URL")
LLM_API_KEY = _endpoint_setting("API_KEY", "LLM_API_KEY")
LLM_STREAMING = os.getenv("LLM_STREAMING", "false").lower() in ("true", "1", "yes")
LLM_REQUEST_TIMEOUT_S = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "600"))
LLM_JSON_REPAIR = os.getenv("LLM_JSON_REPAIR", "false").lower() in ("true", "1", "yes")
OFFICIAL_IO = os.getenv("NLI_OFFICIAL_IO", "false").lower() in ("true", "1", "yes")

# Retries on transient API errors (timeouts, 429, 5xx): exponential backoff within a time budget per request.
# A failed request would otherwise silently count as Neutral
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))
LLM_RETRY_BUDGET_S = float(os.getenv("LLM_RETRY_BUDGET_S", "120"))

# Context strategy: 'hybrid' (booklet-wide BM25 claim + vote title), 'retrieval' (proposal filter + BM25) or 'full'
DEFAULT_STRATEGY = os.getenv("NLI_STRATEGY", "hybrid")
# 12 pages (with PAGE_MAX_CHARS clipping): all 1,495 OST pairs 0.912 -> 0.931 macro-F1 vs. 10 pages, +~890 input tokens
DEFAULT_TOP_K = int(os.getenv("NLI_TOP_K", "12"))
# Max characters per booklet passage; 0 = whole pages (default: on dev, n=450, pages scored 0.908 F1 vs 0.831 for 600-char passages)
PASSAGE_CHARS = int(os.getenv("PASSAGE_CHARS", "0"))

# Max characters per retrieved page sent to the model (0 = no limit). Longer pages keep the window of
# sentences that best matches claim + vote title; only a few dense legal-text pages are affected.
# 3000: test input tokens 5,725 -> 4,696 (p95 12.7k -> 7.4k), macro-F1 0.925 -> 0.928 (dev 0.908 -> 0.913)
PAGE_MAX_CHARS = int(os.getenv("PAGE_MAX_CHARS", "3000"))

# Prompt mode:
#   'ids'     - JSON with confidences + cited passage ids (default: best F1, ~55 output tokens)
#   'json'    - JSON with reasoning + verbatim quotes (~150 output tokens)
#   'compact' - label digit + ids with logprob confidences (fewest tokens, weaker on Neutral)
PROMPT_MODE = os.getenv("PROMPT_MODE", "ids")

# 'ids' mode: let the model write one short sentence before its confidences (a minimal reasoning step)
IDS_REASON = os.getenv("IDS_REASON", "false").lower() in ("true", "1", "yes")

# 'ids' mode: add worked examples for claims attributed to one side of the booklet
FEW_SHOT = os.getenv("FEW_SHOT", "false").lower() in ("true", "1", "yes")

# 'ids' mode: let the thinking model reason step by step before its JSON answer (much slower, more output tokens)
THINKING = os.getenv("THINKING", "false").lower() in ("true", "1", "yes")
THINKING_MAX_TOKENS = int(os.getenv("THINKING_MAX_TOKENS", "1500"))
# Hard reasoning budget in tokens (0 = none): when it runs out, the reasoning is closed and the answer forced
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))

# Cross-lingual pairs: translate the claim into the booklet language first (one extra short call) and give
# the model both the original and the translation
TRANSLATE_CLAIM = os.getenv("TRANSLATE_CLAIM", "false").lower() in ("true", "1", "yes")

# Speaker-aware retrieval: hide the opposing side's argument pages when a claim names who says something
SPEAKER_AWARE = os.getenv("SPEAKER_AWARE", "false").lower() in ("true", "1", "yes")

# Speaker boost: if a claim names one side, its N best argument pages are always among the retrieved pages
# (replacing the lowest-ranked other pages, so the page count stays NLI_TOP_K). Gold recall of committee claims
# 0.888 -> 0.985 with N=2 on all 1,495 pairs; macro-F1 0.931 -> 0.940 (22 fixed / 8 broken), no extra tokens
SPEAKER_BOOST = int(os.getenv("SPEAKER_BOOST", "2"))
# Speaker hint: tell the model which passages are the named side's own text
SPEAKER_HINT = os.getenv("SPEAKER_HINT", "false").lower() in ("true", "1", "yes")

# Hard override of the model label on a detected numerical clash.
# Off by default: on the OST benchmark it flipped correct Neutral predictions to Contradiction
# (claims citing a year the booklet never mentions are Neutral, not contradicted).
NUMERIC_OVERRIDE = os.getenv("NUMERIC_OVERRIDE", "false").lower() in ("true", "1", "yes")

# Independent evidence postprocessing; 'legacy' preserves the baseline policy.
EVIDENCE_POLICY = os.getenv("EVIDENCE_POLICY", "legacy")
if EVIDENCE_POLICY not in ("legacy", "raw_pages", "raw_pages_and_blocks"):
    raise ValueError("Unsupported EVIDENCE_POLICY")
CACHE_SINGLE_FLIGHT = os.getenv("CACHE_SINGLE_FLIGHT", "false").lower() in ("true", "1", "yes")
SOURCE_PAGES_LAZY = os.getenv("SOURCE_PAGES_LAZY", "false").lower() in ("true", "1", "yes")
RETRIEVAL_QUERY_MODE = os.getenv("RETRIEVAL_QUERY_MODE", "original")
if RETRIEVAL_QUERY_MODE not in ("original", "translated", "union"):
    raise ValueError("Unsupported RETRIEVAL_QUERY_MODE")
QUERY_TRANSLATION_CACHE_DIR = Path(os.getenv("QUERY_TRANSLATION_CACHE_DIR", str(Path(tempfile.gettempdir()) / "fact-attack" / "queries")))

# Mock Mode (useful for offline testing or prior to receiving CSCS key)
MOCK_APERTUS = os.getenv("MOCK_APERTUS", "false").lower() in ("true", "1", "yes")

# Challenge Label Mappings
LABEL_MAPPING = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction",
}

LABEL_EXPLANATIONS = {
    0: "The booklet supports the statement.",
    1: "The booklet does not provide enough information either way.",
    2: "The booklet contradicts the statement.",
}

# Parsed booklets are cached on disk, keyed by file content (so a re-mounted PDF path still hits the cache)
BOOKLET_CACHE_DIR = Path(os.getenv("BOOKLET_CACHE_DIR", str(Path(tempfile.gettempdir()) / "fact-attack" / "booklets")))

# Directories
DATA_DIR = BASE_DIR / "data"
BOOKLETS_DIR = DATA_DIR / "booklets"
BENCHMARK_PATH = DATA_DIR / "demo_dataset.jsonl"
