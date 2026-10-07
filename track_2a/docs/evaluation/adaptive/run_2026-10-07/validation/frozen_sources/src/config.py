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

# Existing default is retained until a measured dense-retrieval advantage exists.
STRATEGIES = ("retrieval", "full", "hybrid", "hybrid_dense")
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

# Optional local embeddings. No model is imported/downloaded by configuration.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION", "5617a9f61b028005a4858fdac845db406aefb181")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "2048"))
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() in ("true", "1", "yes")
EMBEDDING_CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", str(BASE_DIR / ".cache" / "embeddings"))).expanduser()
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "40"))
HYBRID_FINAL_PAGES = int(os.getenv("HYBRID_FINAL_PAGES", "10"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_DENSE_FALLBACK = os.getenv("HYBRID_DENSE_FALLBACK", "false").lower() in ("true", "1", "yes")
