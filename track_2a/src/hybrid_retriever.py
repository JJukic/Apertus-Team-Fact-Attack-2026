"""Unfiltered paragraph search, max-per-page aggregation, deterministic page RRF."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings import DenseIndex, EmbeddingError, HybridSettings
from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever

logger = logging.getLogger(__name__)


def normalize_scores(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or np.ptp(scores) == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / np.ptp(scores)


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60):
    if k < 1:
        raise ValueError("RRF constant must be positive")
    scores = {}
    for ranking in rankings:
        # Duplicates contribute only once and do not consume another rank.
        for rank, page in enumerate(dict.fromkeys(ranking), 1):
            scores[page] = scores.get(page, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda page: (-scores[page], page)), scores


class HybridPageRetriever:
    def __init__(self, pages: List[Dict[str, Any]], paragraphs: List[Dict[str, Any]],
                 source_hash: str, settings: Optional[HybridSettings] = None,
                 backend: Optional[Any] = None):
        self.settings = settings or HybridSettings()
        self.source_hash = source_hash
        self.pages = {}
        for page in pages:
            number = page["page_number"]
            if number in self.pages and page != self.pages[number]:
                raise ValueError(f"Conflicting original page objects for page {number}")
            self.pages[number] = page.copy()
        self.paragraphs = [p for p in paragraphs if p["page_number"] in self.pages and p["text"].strip()]
        tokens = [PassageRetriever._tokenize(p["text"]) for p in self.paragraphs]
        self.bm25 = BM25Okapi(tokens) if tokens and any(tokens) else None
        self.index = DenseIndex(source_hash, self.paragraphs, PDFParser.VERSION, self.settings, backend)

    def _page_ranking(self, scores) -> List[int]:
        maxima = {}
        for para, score in zip(self.paragraphs, scores):
            page = para["page_number"]
            maxima[page] = max(float(score), maxima.get(page, float("-inf")))
        # Max, never sum: many paragraphs cannot accumulate extra page votes.
        return sorted(maxima, key=lambda page: (-maxima[page], page))[:self.settings.candidates]

    def lexical_ranking(self, claim: str, vote: Optional[str] = None) -> List[int]:
        if self.bm25 is None:
            return []
        claim_scores = normalize_scores(self.bm25.get_scores(PassageRetriever._tokenize(claim)))
        title_scores = normalize_scores(self.bm25.get_scores(PassageRetriever._tokenize(vote or "")))
        scores = claim_scores + 2.0 * title_scores
        # A no-signal list must not inject arbitrary early pages into cross-language RRF.
        return self._page_ranking(scores) if np.any(scores) else []

    def retrieve(self, claim: str, vote: Optional[str] = None, dense: bool = True):
        if not claim.strip():
            raise ValueError("Claim must not be empty")
        if not self.paragraphs:
            raise EmbeddingError("PDF has no extractable paragraphs; OCR may be needed")
        start = time.perf_counter()
        lexical = self.lexical_ranking(claim, vote)
        metrics = {"requested_strategy": "hybrid_dense" if dense else "hybrid",
                   "actual_strategy": "hybrid_dense" if dense else "hybrid",
                   "cache_status": "not_used", "embedding_ms": 0.0, "model_load_ms": 0.0,
                   "query_embedding_ms": 0.0, "document_embedding_ms": 0.0,
                   "index_build_ms": 0.0, "index_load_ms": 0.0}
        ranked, scores = lexical, {}
        if dense:
            try:
                cosine, stats = self.index.search(claim)
                metrics.update(stats)
                ranked, scores = reciprocal_rank_fusion(
                    [lexical, self._page_ranking(cosine)], self.settings.rrf_k,
                )
            except EmbeddingError as exc:
                if not self.settings.fallback:
                    raise
                logger.warning("hybrid_dense failed; actual strategy=hybrid (explicit fallback): %s", exc)
                metrics.update(actual_strategy="hybrid", cache_status="error", fallback_reason=str(exc))
        selected = []
        for number in ranked[:self.settings.final_pages]:
            page = self.pages[number].copy()
            paras = [p for p in self.paragraphs if p["page_number"] == number]
            page["source_id"] = f"{self.source_hash}:page:{number}"
            # Preserve original page text/metadata and available parser annotations.
            for field, plural in (("section_type", "section_types"), ("proposal_id", "proposal_ids"),
                                  ("speaker", "speakers")):
                values = list(dict.fromkeys(p[field] for p in paras if p.get(field) is not None))
                if values:
                    page[plural] = values
                    if len(values) == 1:
                        page.setdefault(field, values[0])
            page["retrieval_score"] = scores.get(number)
            selected.append(page)
        metrics["selected_pages"] = [p["page_number"] for p in selected]
        metrics["retrieval_ms"] = (time.perf_counter() - start) * 1000
        return selected, metrics

    def warm_up(self):
        # Even a disk hit must load the model once, to warm query encoding.
        stats = self.index.ensure()
        if hasattr(self.index.backend, "load"):
            self.index.backend.load()
            stats["model_load_ms"] += self.index.backend.last_load_ms
        return dict(stats, cache_key=self.index.key, paragraph_count=len(self.paragraphs),
                    page_count=len(self.pages), model=self.settings.model,
                    revision=self.settings.revision, device=self.settings.device)


def page_context(pages: List[Dict[str, Any]]) -> str:
    blocks = []
    for page in pages:
        labels = [f"[Page {page['page_number']}]"]
        for field, label in (("section_types", "Section"), ("proposal_ids", "Proposal"), ("speakers", "Speaker")):
            if page.get(field):
                labels.append(f"[{label} {', '.join(str(v) for v in page[field])}]")
        blocks.append(" ".join(labels) + "\n" + page["text"])
    return "\n\n".join(blocks)
