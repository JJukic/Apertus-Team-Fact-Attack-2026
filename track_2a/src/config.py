"""
Configuration and Environment Management for Hack Apertus Track 2A (OST).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for local configuration in track_2a first, then the repo root.
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# Local files take precedence over .env; existing environment variables are kept.
env_paths = [
    BASE_DIR / ".env.local",
    BASE_DIR / ".env",
    REPO_ROOT / ".env.local",
    REPO_ROOT / ".env",
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

# Apertus / LLM Endpoint Config
LLM_NAME = os.getenv("LLM_NAME", "swiss-ai/Apertus-v1.5-8B")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.inference.cscs.ch/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Execution Strategy: 'retrieval' or 'full'
DEFAULT_STRATEGY = os.getenv("NLI_STRATEGY", "retrieval")

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
