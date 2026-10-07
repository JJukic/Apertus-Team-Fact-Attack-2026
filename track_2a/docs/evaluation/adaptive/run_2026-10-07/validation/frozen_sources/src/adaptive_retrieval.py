"""Independent candidate generators, non-lossy union and NLI-aware reranking."""
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
import time
import hashlib
import json
import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings import DenseIndex, HybridSettings
from src.evidence_units import EvidenceUnitParser, expand_neighbors


@dataclass(frozen=True)
class AdaptiveSettings:
    candidate_k: int = 20
    evidence_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    translated_bm25: bool = True
    dense: bool = True
    ranking: str = "nli"
    neighbor_radius: int = 1
    reranker_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    reranker_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    reranker_max_length: int = 512
    reranker_stride: int = 64
    reranker_batch_size: int = 4
    device: str = "cpu"
    local_files_only: bool = True

    def __post_init__(self):
        if self.candidate_k < 1 or self.evidence_k < 1 or self.neighbor_radius < 0:
            raise ValueError("Positive candidate/evidence counts and nonnegative neighbors required")
        if self.ranking not in ("union", "bm25", "dense", "rrf", "nli"):
            raise ValueError("Unknown ranking method")
        if self.ranking == "dense" and not self.dense:
            raise ValueError("Dense ranking requires the dense candidate generator")
        if not 0 <= self.bm25_b <= 1 or self.bm25_k1 <= 0:
            raise ValueError("Invalid BM25 parameters")
        if not re.fullmatch(r"[a-f0-9]{40}", self.reranker_revision):
            raise ValueError("Reranker revision must be an immutable model commit")
        if not 0 <= self.reranker_stride < self.reranker_max_length-32 or self.reranker_batch_size < 1:
            raise ValueError("Invalid reranker window/batch settings")


@dataclass
class Candidate:
    unit: object
    ranks: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    nli_probabilities: dict = field(default_factory=dict)
    reranker_score: float = 0.0
    windows: int = 0

    def to_dict(self):
        return {"unit": self.unit.to_dict(), "ranks": self.ranks, "scores": self.scores,
                "nli_probabilities": self.nli_probabilities,
                "reranker_score": self.reranker_score, "windows": self.windows}


def tokenize(text):
    # Normalize PDF line-end word breaks for search only; original units stay intact.
    searchable = re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", text)
    return re.findall(r"\w+", searchable.lower(), re.UNICODE)


class BM25Generator:
    def __init__(self, units, *, k1=1.5, b=0.75):
        self.units = units
        self.model = BM25Okapi([tokenize(u.text) for u in units], k1=k1, b=b) if units else None

    def retrieve(self, query, top_k=20):
        if self.model is None:
            return []
        scores = self.model.get_scores(tokenize(query))
        order = sorted(range(len(self.units)), key=lambda i: (-float(scores[i]), self.units[i].order))
        return [(self.units[i], float(scores[i])) for i in order[:top_k]]


class DenseGenerator:
    def __init__(self, units, settings=None, backend=None):
        if not units:
            raise ValueError("Dense retrieval needs source evidence units")
        self.units = units
        self.index = DenseIndex(units[0].document_id, [u.to_dict() for u in units],
                                EvidenceUnitParser.VERSION, settings or HybridSettings(), backend)
        self.last_metrics = {}

    def retrieve(self, query, top_k=20):
        scores, self.last_metrics = self.index.search(query)
        order = sorted(range(len(self.units)), key=lambda i: (-float(scores[i]), self.units[i].order))
        return [(self.units[i], float(scores[i])) for i in order[:top_k]]


def candidate_union(generators):
    """Round-robin union; no candidate cap or incomparable-score fusion."""
    merged = {}
    for rank in range(max((len(values) for values in generators.values()), default=0)):
        for name, values in generators.items():
            if rank >= len(values):
                continue
            unit, score = values[rank]
            if unit.id in merged and merged[unit.id].unit != unit:
                raise ValueError("Evidence ID collision between different sources")
            candidate = merged.setdefault(unit.id, Candidate(unit))
            candidate.ranks[name] = rank+1
            candidate.scores[name] = float(score)
    return list(merged.values())


class NLIReranker:
    """Score evidence decisiveness = p(entailment) + p(contradiction).

    The passage is the premise and the original claim the hypothesis. Long
    units use overlapping model-input windows; output evidence stays whole.
    """
    def __init__(self, settings):
        self.settings = settings
        self.model = self.tokenizer = None
        self.cache = {}

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        s = self.settings
        self.tokenizer = AutoTokenizer.from_pretrained(s.reranker_model, revision=s.reranker_revision,
            local_files_only=s.local_files_only, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(s.reranker_model,
            revision=s.reranker_revision, local_files_only=s.local_files_only,
            trust_remote_code=False).to(s.device).eval()
        mapping = {str(v).lower(): int(k) for k,v in self.model.config.id2label.items()}
        if set(mapping) != {"entailment", "neutral", "contradiction"}:
            raise ValueError("Reranker must explicitly identify all three NLI labels")
        self.label_indices = mapping

    def score(self, claim, units):
        import torch
        self.load()
        s = self.settings
        keys, pending, windows = [], {}, []
        for unit in units:
            key = (claim, unit.id, hashlib.sha256(unit.text.encode()).hexdigest())
            keys.append(key)
            if key in self.cache or key in pending:
                continue
            encoded = self.tokenizer(unit.text, claim, truncation="only_first",
                max_length=s.reranker_max_length, stride=s.reranker_stride,
                return_overflowing_tokens=True, padding=False)
            encoded.pop("overflow_to_sample_mapping", None)
            count = len(encoded["input_ids"])
            pending[key] = []
            for i in range(count):
                windows.append((key, {name:values[i] for name,values in encoded.items()}))
        # Batch across source units as well as overflow windows. Each score still
        # belongs to one whole original unit and uses its most decisive window.
        for offset in range(0, len(windows), s.reranker_batch_size):
            subset = windows[offset:offset+s.reranker_batch_size]
            batch = self.tokenizer.pad([encoded for _,encoded in subset],
                                       padding=True, return_tensors="pt")
            batch = {k:v.to(s.device) for k,v in batch.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(**batch).logits, dim=-1).cpu().tolist()
            for (key,_), values in zip(subset,probabilities):
                pending[key].append(values)
        for key, probabilities in pending.items():
            window = max(probabilities, key=lambda p: 1-p[self.label_indices["neutral"]])
            values = {label:float(window[index]) for label,index in self.label_indices.items()}
            self.cache[key] = {"probabilities":values,
                "score":values["entailment"]+values["contradiction"], "windows":len(probabilities)}
        return [self.cache[key] for key in keys]

    def rerank(self, claim, candidates):
        scores = self.score(claim, [c.unit for c in candidates])
        for candidate, values in zip(candidates, scores):
            candidate.nli_probabilities = values["probabilities"]
            candidate.reranker_score = values["score"]
            candidate.windows = values["windows"]
        return sorted(candidates, key=lambda c: (-c.reranker_score, c.unit.order))


class AdaptiveRetriever:
    def __init__(self, units, settings=None, *, dense_settings=None, dense_backend=None, reranker=None,
                 dense_generator=None):
        self.units = units
        self.settings = settings or AdaptiveSettings()
        self.bm25 = BM25Generator(units, k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        self.dense_settings, self.dense_backend = dense_settings, dense_backend
        self.dense_generator = dense_generator
        self.reranker = reranker or NLIReranker(self.settings)

    def retrieve(self, claim, queries, booklet_language):
        start = time.perf_counter()
        s = self.settings
        tasks = {"bm25_original": (self.bm25.retrieve, claim)}
        if s.translated_bm25 and queries[booklet_language] != claim:
            tasks["bm25_translated"] = (self.bm25.retrieve, queries[booklet_language])
        if s.dense:
            if self.dense_generator is None:
                self.dense_generator = DenseGenerator(self.units, self.dense_settings, self.dense_backend)
            tasks["dense"] = (self.dense_generator.retrieve, claim)
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {name:executor.submit(function, query, s.candidate_k)
                       for name,(function,query) in tasks.items()}
            # Resolve all independent generators; a failed component cannot silently disappear.
            generators = {name:future.result() for name,future in futures.items()}
        candidates = candidate_union(generators)
        reranking_start = time.perf_counter()
        if s.ranking == "nli":
            ranked = self.reranker.rerank(claim, candidates)
        elif s.ranking == "rrf":
            ranked = sorted(candidates, key=lambda c: (-sum(1/(60+r) for r in c.ranks.values()), c.unit.order))
        elif s.ranking in ("bm25", "dense"):
            name = "bm25_original" if s.ranking == "bm25" else "dense"
            ranked = sorted([c for c in candidates if name in c.ranks],
                            key=lambda c: (c.ranks[name], c.unit.order))
        else:
            ranked = candidates
        selected = [c.unit for c in ranked[:s.evidence_k]]
        expanded = expand_neighbors(selected, self.units, s.neighbor_radius)
        return {"generators": {name:[{"id":u.id,"score":score,"rank":i+1}
                                      for i,(u,score) in enumerate(values)] for name,values in generators.items()},
                "candidates": [c.to_dict() for c in candidates],
                "ranked_ids": [c.unit.id for c in ranked], "selected_ids": [u.id for u in selected],
                "expanded_ids": [u.id for u in expanded], "evidence": expanded,
                "signals": {"best_reranker_score": ranked[0].reranker_score if ranked else 0,
                            "reranker_margin": ranked[0].reranker_score-ranked[1].reranker_score if len(ranked)>1 else 0,
                            "retrieval_agreement": len(ranked[0].ranks) if ranked else 0},
                "reranking_ms": (time.perf_counter()-reranking_start)*1000,
                "retrieval_ms": (time.perf_counter()-start)*1000,
                "dense_metrics": self.dense_generator.last_metrics if s.dense else {}}
