"""CLI-compatible adaptive evidence retrieval and strictly grounded Apertus NLI."""
from dataclasses import asdict, replace
from pathlib import Path
import time
import json

from src.embeddings import HybridSettings, pdf_hash
from src.evidence_units import EvidenceUnitParser
from src.adaptive_retrieval import AdaptiveSettings, AdaptiveRetriever, DenseGenerator, NLIReranker
from src.grounded_apertus import ApertusJSONTransport, QueryPreparer, GroundedApertusNLI
from src.adaptive_calibration import NeutralPolicy
from src import config


def experiment_settings(name, *, base=None, neighbor_radius=1):
    base = base or AdaptiveSettings()
    profiles = {
        "E0": dict(translated_bm25=False, dense=False, ranking="bm25", neighbor_radius=0),
        "E1": dict(translated_bm25=False, dense=False, ranking="bm25", neighbor_radius=0),
        "E2": dict(translated_bm25=False, dense=False, ranking="bm25", neighbor_radius=neighbor_radius),
        "E3": dict(translated_bm25=True, dense=False, ranking="union", neighbor_radius=0),
        "E4": dict(translated_bm25=True, dense=True, ranking="union", neighbor_radius=0),
        "E5": dict(translated_bm25=True, dense=True, ranking="nli", neighbor_radius=0),
        "E6": dict(translated_bm25=True, dense=True, ranking="nli", neighbor_radius=neighbor_radius),
        "E7": dict(translated_bm25=True, dense=True, ranking="nli", neighbor_radius=neighbor_radius),
        "E8": dict(translated_bm25=True, dense=True, ranking="nli", neighbor_radius=neighbor_radius),
    }
    if name not in profiles:
        raise ValueError("Unknown ablation; use E0–E8")
    return replace(base, **profiles[name])


class AdaptiveVerificationEngine:
    def __init__(self, *, transport=None, cache_dir=None, dense_settings=None):
        self.transport = transport or ApertusJSONTransport()
        self.cache_dir = Path(cache_dir or config.BASE_DIR / ".cache/adaptive")
        self.queries = QueryPreparer(self.transport, self.cache_dir / "queries")
        self.nli = GroundedApertusNLI(self.transport)
        self.dense_settings = dense_settings or HybridSettings(local_files_only=True)
        self.documents, self.dense, self.rerankers = {}, {}, {}

    def document(self, booklet, language=None):
        key = (pdf_hash(Path(booklet)), language, EvidenceUnitParser.VERSION)
        if key not in self.documents:
            parser = EvidenceUnitParser()
            units = parser.parse(Path(booklet), language)
            self.documents[key] = (units, parser.diagnostics)
        return self.documents[key]

    def prepare(self, claim, booklet, *, claim_language=None, booklet_language=None,
                experiment="E6", settings=None, neighbor_radius=1):
        start = time.perf_counter()
        settings = settings or experiment_settings(experiment, neighbor_radius=neighbor_radius)
        units, parsing = self.document(booklet, booklet_language)
        language = units[0].language
        translate = settings.translated_bm25 and (claim_language is None or claim_language != language)
        queries = self.queries.prepare(claim, claim_language, translate=translate)
        if experiment == "E0":
            retrieved = {"generators": {}, "candidates": [], "ranked_ids": [u.id for u in units],
                "selected_ids": [u.id for u in units], "expanded_ids": [u.id for u in units],
                "evidence": units, "signals": {}, "retrieval_ms": 0, "reranking_ms": 0}
        else:
            key = (units[0].document_id, units[0].language)
            if settings.dense and key not in self.dense:
                self.dense[key] = DenseGenerator(units, self.dense_settings)
            rkey = (settings.reranker_model, settings.reranker_revision, settings.device,
                    settings.reranker_max_length, settings.reranker_stride, settings.local_files_only)
            if rkey not in self.rerankers:
                self.rerankers[rkey] = NLIReranker(settings)
            retriever = AdaptiveRetriever(units, settings, dense_settings=self.dense_settings,
                                          reranker=self.rerankers[rkey], dense_generator=self.dense.get(key))
            retrieved = retriever.retrieve(claim, queries["queries"], language)
        return {"claim": claim, "claim_language": queries["claim_language"],
                "booklet": str(Path(booklet).resolve()),
                "dense_configuration": json.loads(json.dumps(asdict(self.dense_settings), default=str)),
                "booklet_language": language, "experiment": experiment,
                "settings": asdict(settings), "parsing": parsing, "queries": queries,
                "retrieval": retrieved, "preparation_ms": (time.perf_counter()-start)*1000}

    def verify(self, claim, booklet, *, claim_language=None, booklet_language=None,
               experiment="E6", settings=None, neighbor_radius=1, neutral_policy=None,
               case_id="case-0001", prepared=None):
        if experiment == "E7" and neutral_policy is None:
            raise ValueError("E7 requires a policy selected on validation data")
        start = time.perf_counter()
        before = (self.transport.requests, self.transport.input_tokens, self.transport.output_tokens)
        reused_preparation = prepared is not None
        prepared = prepared or self.prepare(claim, booklet, claim_language=claim_language,
            booklet_language=booklet_language, experiment=experiment,
            settings=settings, neighbor_radius=neighbor_radius)
        if prepared["claim"] != claim or prepared["experiment"] != experiment:
            raise ValueError("Frozen preparation belongs to a different claim/experiment")
        if prepared["booklet"] != str(Path(booklet).resolve()) or prepared["parsing"]["document_sha256"] != pdf_hash(Path(booklet)):
            raise ValueError("Frozen preparation belongs to a different or changed booklet")
        if claim_language is not None and prepared["claim_language"] != claim_language:
            raise ValueError("Frozen preparation has a different claim language")
        units = prepared["retrieval"]["evidence"]
        decision = self.nli.classify(claim, units, claim_language=prepared["claim_language"],
                                     mode="binary" if experiment == "E8" else "direct")
        raw_label = decision.label
        if neutral_policy is not None:
            decision.label = neutral_policy.apply(decision.label, decision.probabilities,
                                                   prepared["retrieval"]["signals"])
            if decision.label == 1:
                decision.evidence_ids, decision.evidence = [], []
        retrieval = {key:value for key,value in prepared["retrieval"].items() if key != "evidence"}
        elapsed = (time.perf_counter()-start)*1000
        cost = {"api_requests": self.transport.requests-before[0],
                "input_tokens": self.transport.input_tokens-before[1],
                "output_tokens": self.transport.output_tokens-before[2], "observed_latency_ms": elapsed,
                "preparation_reused": reused_preparation,
                "total_latency_ms": elapsed + (prepared["preparation_ms"] if reused_preparation else 0)}
        return {"id": case_id, "claim": claim, "experiment": experiment,
                "label": decision.label, "raw_label": raw_label,
                "label_name": config.LABEL_MAPPING[decision.label], "decision": decision.to_dict(),
                "booklet": str(booklet), "booklet_sha256": prepared["parsing"]["document_sha256"],
                "claim_language": prepared["claim_language"], "booklet_language": prepared["booklet_language"],
                "configuration": prepared["settings"], "parsing": prepared["parsing"],
                "model": self.transport.model, "dense_configuration": prepared["dense_configuration"],
                "queries": prepared["queries"], "retrieval": retrieval,
                "neutral_policy": neutral_policy.to_dict() if neutral_policy else None, "cost": cost,
                "official": {"id": case_id, "label": decision.label,
                    "label_name": config.LABEL_MAPPING[decision.label].lower(),
                    "evidence": [{"page":e["page"], "text":e["text"]} for e in decision.evidence],
                    "metrics": {"input_tokens": cost["input_tokens"], "output_tokens": cost["output_tokens"],
                                "inference_time_ms": round(elapsed)}}}
