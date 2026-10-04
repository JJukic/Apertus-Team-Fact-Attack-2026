"""
Configuration and Environment Management for Hack Apertus Track 2A (OST).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

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
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.inference.cscs.ch/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Context strategy: 'hybrid' (booklet-wide BM25 claim + vote title), 'retrieval' (proposal filter + BM25) or 'full'
DEFAULT_STRATEGY = os.getenv("NLI_STRATEGY", "hybrid")
DEFAULT_TOP_K = int(os.getenv("NLI_TOP_K", "10"))
# Max characters per booklet passage; 0 = whole pages (default: on dev, n=450, pages scored 0.908 F1 vs 0.831 for 600-char passages)
PASSAGE_CHARS = int(os.getenv("PASSAGE_CHARS", "0"))

# Prompt mode:
#   'ids'     - JSON with confidences + cited passage ids (default: best F1, ~55 output tokens)
#   'json'    - JSON with reasoning + verbatim quotes (~150 output tokens)
#   'compact' - label digit + ids with logprob confidences (fewest tokens, weaker on Neutral)
PROMPT_MODE = os.getenv("PROMPT_MODE", "ids")

# 'ids' mode: let the model write one short sentence before its confidences (a minimal reasoning step)
IDS_REASON = os.getenv("IDS_REASON", "false").lower() in ("true", "1", "yes")

# Hard override of the model label on a detected numerical clash.
# Off by default: on the OST benchmark it flipped correct Neutral predictions to Contradiction
# (claims citing a year the booklet never mentions are Neutral, not contradicted).
NUMERIC_OVERRIDE = os.getenv("NUMERIC_OVERRIDE", "false").lower() in ("true", "1", "yes")

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

# Directories
DATA_DIR = BASE_DIR / "data"
BOOKLETS_DIR = DATA_DIR / "booklets"
BENCHMARK_PATH = DATA_DIR / "demo_dataset.jsonl"
