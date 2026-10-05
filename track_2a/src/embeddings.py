"""Lazy local dense embeddings and a versioned, atomic on-disk paragraph index."""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import zipfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src import config

logger = logging.getLogger(__name__)
_MODEL_LOAD_LOCK = threading.Lock()


class EmbeddingError(RuntimeError):
    """An actionable local model/index error, never an implicit BM25 fallback."""


@dataclass(frozen=True)
class HybridSettings:
    model: str = config.EMBEDDING_MODEL
    revision: str = config.EMBEDDING_REVISION
    device: str = config.EMBEDDING_DEVICE
    batch_size: int = config.EMBEDDING_BATCH_SIZE
    max_length: int = config.EMBEDDING_MAX_LENGTH
    local_files_only: bool = config.EMBEDDING_LOCAL_FILES_ONLY
    cache_dir: Path = config.EMBEDDING_CACHE_DIR
    candidates: int = config.HYBRID_CANDIDATES
    final_pages: int = config.HYBRID_FINAL_PAGES
    rrf_k: int = config.HYBRID_RRF_K
    fallback: bool = config.HYBRID_DENSE_FALLBACK

    def __post_init__(self):
        for name in ("batch_size", "max_length", "candidates", "final_pages", "rrf_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.candidates < self.final_pages:
            raise ValueError("HYBRID_CANDIDATES must be >= HYBRID_FINAL_PAGES")
        if not self.model or not self.revision:
            raise ValueError("Embedding model and immutable revision must be configured")


def pdf_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as pdf:
        for block in iter(lambda: pdf.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_embeddings(values: Any, rows: int) -> np.ndarray:
    """Validate and L2-normalize; zero/missing/nonfinite vectors are errors."""
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise EmbeddingError("Embedding output is not a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise EmbeddingError(f"Expected {rows} embedding rows, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise EmbeddingError("Embedding output contains nonfinite or zero vectors")
    return matrix / norms


def _load_model(model: str, revision: str, device: str, max_length: int, local_only: bool):
    # lru_cache alone can execute duplicate cold loads in concurrent threads.
    with _MODEL_LOAD_LOCK:
        return _cached_model(model, revision, device, max_length, local_only)


@lru_cache(maxsize=2)
def _cached_model(model: str, revision: str, device: str, max_length: int, local_only: bool):
    # Heavy libraries and weights are loaded only from the dense strategy/warm-up.
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise EmbeddingError("EMBEDDING_REVISION must be an immutable 40-character model commit, not a moving branch")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError("Dense retrieval needs: pip install -r requirements-hybrid.txt") from exc
    try:
        instance = SentenceTransformer(
            model, revision=revision, device=device, local_files_only=local_only,
            trust_remote_code=False,
        )
        instance.max_seq_length = max_length
        return instance
    except Exception as exc:
        raise EmbeddingError(
            f"Cannot load embedding model {model}@{revision}. Check weights, network, "
            "EMBEDDING_LOCAL_FILES_ONLY and device; run 'python -m src index' to warm up."
        ) from exc


class SentenceTransformerBackend:
    """One reusable local model; BGE-M3 requires no query instruction prefix."""

    def __init__(self, settings: HybridSettings):
        self.settings = settings
        self.model = None
        self.last_load_ms = 0.0

    def load(self):
        self.last_load_ms = 0.0
        if self.model is None:
            start = time.perf_counter()
            s = self.settings
            self.model = _load_model(s.model, s.revision, s.device, s.max_length, s.local_files_only)
            self.last_load_ms = (time.perf_counter() - start) * 1000

    def encode(self, texts: List[str]) -> np.ndarray:
        self.load()
        try:
            return self.model.encode(
                texts, batch_size=self.settings.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Local embedding computation failed; check device memory/batch size") from exc


class DenseIndex:
    VERSION = "dense-index-v1-l2-float32"

    def __init__(self, source_hash: str, paragraphs: List[Dict[str, Any]],
                 parser_version: str, settings: HybridSettings, backend: Optional[Any] = None):
        self.paragraphs = paragraphs
        self.settings = settings
        self.backend = backend if backend is not None else SentenceTransformerBackend(settings)
        self.vectors = None
        self._lock = threading.RLock()
        chunks = json.dumps(paragraphs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.metadata = {
            "pdf_sha256": source_hash, "parser_version": parser_version,
            "chunks_sha256": hashlib.sha256(chunks.encode()).hexdigest(),
            "model": settings.model, "revision": settings.revision,
            "max_length": settings.max_length, "index_version": self.VERSION,
        }
        payload = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        self.key = hashlib.sha256(payload.encode()).hexdigest()
        self.path = Path(settings.cache_dir) / f"{self.key}.npz"

    @staticmethod
    def _checksum(matrix: np.ndarray) -> str:
        return hashlib.sha256(matrix.tobytes()).hexdigest()

    def ensure(self) -> Dict[str, Any]:
        # Concurrent sessions sharing an engine must not encode the PDF twice.
        with self._lock:
            return self._ensure()

    def _ensure(self) -> Dict[str, Any]:
        if self.vectors is not None:
            return {"cache_status": "memory_hit", "index_build_ms": 0.0,
                    "index_load_ms": 0.0, "document_embedding_ms": 0.0, "model_load_ms": 0.0}
        if not self.paragraphs:
            raise EmbeddingError("PDF has no extractable paragraphs; OCR may be needed")
        start = time.perf_counter()
        status = "miss"
        if self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as cached:
                    metadata = json.loads(str(cached["metadata"].item()))
                    matrix = cached["vectors"]
                    checksum = str(cached["checksum"].item())
                if metadata != self.metadata or checksum != self._checksum(matrix):
                    raise ValueError("Cache metadata/checksum mismatch")
                normalized = normalize_embeddings(matrix, len(self.paragraphs))
                if not np.allclose(matrix, normalized, atol=1e-5):
                    raise ValueError("Cache vectors are not normalized")
                self.vectors = matrix
                return {"cache_status": "disk_hit", "index_build_ms": 0.0,
                        "index_load_ms": (time.perf_counter() - start) * 1000,
                        "document_embedding_ms": 0.0, "model_load_ms": 0.0}
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, EmbeddingError) as exc:
                logger.warning("Rebuilding corrupt embedding cache %s: %s", self.path.name, exc)
                status = "rebuilt_corrupt"
        load_ms = (time.perf_counter() - start) * 1000
        build_start = time.perf_counter()
        self.vectors = self.encode([p["text"] for p in self.paragraphs])
        model_ms = float(getattr(self.backend, "last_load_ms", 0.0))
        encode_ms = (time.perf_counter() - build_start) * 1000 - model_ms
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".npz", delete=False) as temp:
                temp_path = Path(temp.name)
                np.savez_compressed(temp, metadata=json.dumps(self.metadata, sort_keys=True),
                                    vectors=self.vectors, checksum=self._checksum(self.vectors))
            try:
                os.replace(temp_path, self.path)
            finally:
                temp_path.unlink(missing_ok=True)
        except OSError as exc:
            self.vectors = None
            raise EmbeddingError(f"Cannot write embedding cache at {self.path.parent}") from exc
        return {"cache_status": status, "index_build_ms": (time.perf_counter() - build_start) * 1000,
                "index_load_ms": load_ms, "document_embedding_ms": max(0.0, encode_ms),
                "model_load_ms": model_ms}

    def encode(self, texts: List[str]) -> np.ndarray:
        try:
            values = self.backend.encode(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding backend failed to return vectors") from exc
        return normalize_embeddings(values, len(texts))

    def search(self, claim: str):
        with self._lock:
            return self._search(claim)

    def _search(self, claim: str):
        stats = self.ensure()
        start = time.perf_counter()
        query = self.encode([claim])
        load_ms = float(getattr(self.backend, "last_load_ms", 0.0))
        stats["model_load_ms"] += load_ms
        stats["query_embedding_ms"] = max(0.0, (time.perf_counter() - start) * 1000 - load_ms)
        stats["embedding_ms"] = stats["document_embedding_ms"] + stats["query_embedding_ms"]
        if query.shape[1] != self.vectors.shape[1]:
            raise EmbeddingError("Query and cached document embedding dimensions differ; reindex the booklet")
        # L2-normalized dot product equals cosine similarity. Exact local search.
        return self.vectors @ query[0], stats
