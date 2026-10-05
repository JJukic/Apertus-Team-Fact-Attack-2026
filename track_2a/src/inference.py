"""
Inference Pipeline for Claim Verification over Swiss Voting Booklets.
Coordinates PDF parsing, passage retrieval, numerical conflict detection,
page-attributed evidence extraction, and Apertus LLM inference.
"""

import re
import time
import threading
from pypdf.errors import PyPdfError
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient, NLIOutput
from src.numerical_checker import detect_numerical_conflict, NumericalConflictResult
from src.embeddings import HybridSettings, pdf_hash
from src.hybrid_retriever import HybridPageRetriever, page_context
from src import config


class EvidenceSource(BaseModel):
    quote: str
    page_number: Optional[int] = None
    proposal_id: Optional[int] = None
    section_type: Optional[str] = None
    speaker: Optional[str] = None


class PredictionResult(BaseModel):
    id: Optional[str] = "case-0001"
    claim: str
    label: int
    label_name: str
    reasoning: str
    evidence: List[str]
    evidence_sources: List[EvidenceSource] = Field(default_factory=list)
    numerical_conflict: Optional[str] = None
    p_entail: float = 0.0
    p_neutral: float = 0.0
    p_contra: float = 0.0
    fuzzy_rule: Optional[str] = None
    decision_rule: Optional[str] = None
    strategy: str
    booklet_path: str
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    latency_ms: float
    # API latency above keeps its original meaning; these fields are additive.
    total_latency_ms: float = 0.0
    retrieval_metrics: Dict[str, Any] = Field(default_factory=dict)
    requested_strategy: Optional[str] = None

    def to_official_dict(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export output strictly conforming to the Hack Apertus Track 2A (OST) JSON Schema.
        For Neutral (1), evidence is strictly an empty list [].
        """
        cid = case_id or self.id or "case-0001"
        ev_list: List[Dict[str, Any]] = []

        if self.label != 1:  # Evidence is only valid for Entailment (0) and Contradiction (2)
            if self.evidence_sources:
                for src in self.evidence_sources:
                    clean_text = re.sub(r"^\[(?:page|seite)\s+\d+\]\s*\d*\s*", "", src.quote, flags=re.IGNORECASE).strip(' "«»')
                    if clean_text:
                        ev_list.append({
                            "page": src.page_number or 1,
                            "text": clean_text,
                        })
            elif self.evidence:
                for ev in self.evidence:
                    clean_text = re.sub(r"^\[(?:page|seite)\s+\d+\]\s*\d*\s*", "", ev, flags=re.IGNORECASE).strip(' "«»')
                    if clean_text:
                        ev_list.append({
                            "page": 1,
                            "text": clean_text,
                        })

        return {
            "id": cid,
            "label": self.label,
            "label_name": self.label_name.lower(),
            "evidence": ev_list,
            "metrics": {
                "input_tokens": int(self.tokens_prompt),
                "output_tokens": int(self.tokens_completion),
                "inference_time_ms": int(round(self.latency_ms)),
            },
        }


_GLOBAL_BOOKLET_CACHE: Dict[str, Dict[str, Any]] = {}


class ClaimVerificationEngine:
    def __init__(
        self,
        strategy: str = config.DEFAULT_STRATEGY,
        apertus_client: Optional[ApertusClient] = None,
        hybrid_settings: Optional[HybridSettings] = None,
        embedding_backend: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.client = apertus_client or ApertusClient()
        self.pdf_parser = PDFParser()
        self.hybrid_settings = hybrid_settings or HybridSettings()
        self.embedding_backend = embedding_backend
        self._hybrid_retrievers: Dict[str, HybridPageRetriever] = {}
        self._hybrid_lock = threading.Lock()

    def _get_booklet_data(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        path_str = str(Path(pdf_path).resolve())
        source_hash = pdf_hash(Path(path_str))
        key = f"{path_str}:{source_hash}:{self.pdf_parser.VERSION}"
        if key not in _GLOBAL_BOOKLET_CACHE:
            try:
                pages = self.pdf_parser.extract_pages(path_str)
                paragraphs = self.pdf_parser.extract_paragraphs(path_str)
                full_text = self.pdf_parser.extract_full_text(path_str)
            except PyPdfError as exc:
                raise ValueError("PDF cannot be read. Download it again or provide a valid, text-readable PDF.") from exc
            retriever = PassageRetriever(paragraphs)
            # Evict older versions of the same path (e.g. a replaced uploaded PDF).
            for old_key in list(_GLOBAL_BOOKLET_CACHE):
                if old_key.startswith(path_str + ":"):
                    del _GLOBAL_BOOKLET_CACHE[old_key]
            _GLOBAL_BOOKLET_CACHE[key] = {
                "pages": pages,
                "paragraphs": paragraphs,
                "full_text": full_text,
                "retriever": retriever,
                "source_hash": source_hash,
            }
        return _GLOBAL_BOOKLET_CACHE[key]

    def _hybrid_retriever(self, data: Dict[str, Any]) -> HybridPageRetriever:
        key = data["source_hash"]
        with self._hybrid_lock:
            if key not in self._hybrid_retrievers:
                self._hybrid_retrievers[key] = HybridPageRetriever(
                    data["pages"], data["paragraphs"], key, self.hybrid_settings, self.embedding_backend,
                )
            return self._hybrid_retrievers[key]

    def warm_up(self, booklet_pdf: Union[str, Path]) -> Dict[str, Any]:
        """Parse, load model, and build/load index without an Apertus request."""
        start = time.perf_counter()
        data = self._get_booklet_data(booklet_pdf)
        stats = self._hybrid_retriever(data).warm_up()
        stats["total_latency_ms"] = (time.perf_counter() - start) * 1000
        return stats

    def retrieve_context(self, claim: str, booklet_pdf: Union[str, Path],
                         strategy: Optional[str] = None, top_k: int = 5,
                         vote: Optional[str] = None) -> Dict[str, Any]:
        """Prepare only document context; no references/labels or generative calls."""
        strat = strategy or self.strategy
        if strat not in config.STRATEGIES:
            raise ValueError(f"Unknown strategy {strat!r}; choose {', '.join(config.STRATEGIES)}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        start = time.perf_counter()
        data = self._get_booklet_data(booklet_pdf)
        preparation_ms = (time.perf_counter() - start) * 1000
        retrieval_start = time.perf_counter()
        metrics = {"requested_strategy": strat, "actual_strategy": strat,
                   "cache_status": "not_used", "embedding_ms": 0.0, "model_load_ms": 0.0,
                   "query_embedding_ms": 0.0, "document_embedding_ms": 0.0,
                   "index_build_ms": 0.0, "index_load_ms": 0.0}
        if strat == "full":
            context, sources = data["full_text"], data["paragraphs"]
        elif strat == "retrieval":
            sources = data["retriever"].retrieve(claim, top_k=top_k, target_vote=vote)
            context = "\n\n".join(f"[Page {p['page_number']}] {p['text']}" for p in sources)
        else:
            sources, stats = self._hybrid_retriever(data).retrieve(claim, vote, dense=strat == "hybrid_dense")
            context = page_context(sources)
            metrics.update(stats)
        metrics["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
        metrics["document_preparation_ms"] = preparation_ms
        metrics["selected_pages"] = list(dict.fromkeys(p["page_number"] for p in sources))
        return {"context": context, "sources": sources, "metrics": metrics, "booklet_data": data}

    def verify_claim(
        self,
        claim: str,
        booklet_pdf: Union[str, Path],
        claim_language: Optional[str] = None,
        strategy: Optional[str] = None,
        top_k: int = 5,
        vote: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> PredictionResult:
        """
        Verify whether the voting booklet entails, contradicts, or is neutral to the claim.
        Supports ADVANCED TASK from Hack Apertus: accepts optional 'vote' title to scope proposal.
        """
        total_start = time.perf_counter()
        prepared = self.retrieve_context(claim, booklet_pdf, strategy, top_k, vote)
        metrics = prepared["metrics"]
        strat = metrics["actual_strategy"]
        context = prepared["context"]
        candidate_paras = prepared["sources"]
        booklet_data = prepared["booklet_data"]

        # 1. Deterministic Numerical Conflict Check
        num_conflict: Optional[NumericalConflictResult] = detect_numerical_conflict(claim, context)
        numerical_conflict_msg = num_conflict.explanation if num_conflict else None

        # 2. Apertus Model Inference
        nli_output: NLIOutput = self.client.infer(
            context=context,
            claim=claim,
            claim_language=claim_language,
        )

        final_label = nli_output.label
        decision_rule = nli_output.decision_rule_applied or nli_output.fuzzy_rule_applied
        p_entail = nli_output.p_entail
        p_contra = nli_output.p_contra
        p_neutral = nli_output.p_neutral

        # If deterministic numerical conflict was detected, override to Contradiction (2)
        if num_conflict and final_label != 2:
            final_label = 2
            p_contra = max(0.95, p_contra)
            decision_rule = f"Decision-Rule 1b: Numerical Clash ({num_conflict.claim_entity} vs {num_conflict.booklet_entity})"

        # Vacuity Guardrail (Formal Verification Principle):
        # A claim cannot be an Entailment (0) if no valid supporting evidence passage was cited from the booklet.
        if final_label == 0:
            has_valid_evidence = any(len(ev.strip()) >= 15 for ev in nli_output.evidence)
            if not has_valid_evidence:
                final_label = 1
                p_neutral = max(0.85, p_neutral)
                p_entail = 0.10
                decision_rule = "Vacuity Guardrail: Ungrounded entailment prevented (no cited evidence) -> Neutral (1)"

        label_name = config.LABEL_MAPPING.get(final_label, "Unknown")

        # 3. Match evidence quotes to exact source page numbers & proposals
        # For Neutral (1), evidence is strictly an empty list []
        evidence_sources: List[EvidenceSource] = []
        final_evidence_list: List[str] = []

        if final_label != 1:
            all_paras = booklet_data["paragraphs"]
            for ev in nli_output.evidence:
                if metrics["requested_strategy"] in ("hybrid", "hybrid_dense"):
                    # Original page numbers are the prompt IDs; never renumber by rank.
                    clean = re.sub(r"\[(?:page|seite)\s+\d+\]", "", ev, flags=re.IGNORECASE)
                    clean = re.sub(r"^\s*(?:\[(?:Section|Proposal|Speaker)\s+[^\]]*\]\s*)+", "", clean)
                    clean = " ".join(clean.strip(' \"«»').split())
                    quote_pattern = re.compile(re.escape(clean), re.IGNORECASE) if clean else None
                    matches = [p for p in candidate_paras if clean and
                               quote_pattern.search(" ".join(p["text"].split()))]
                    tag = re.search(r"\[(?:page|seite)\s+(\d+)\]", ev, re.IGNORECASE)
                    if tag:
                        tagged = [p for p in matches if p["page_number"] == int(tag.group(1))]
                        matches = tagged or matches
                    if matches:
                        page = matches[0]
                        original = " ".join(page["text"].split())
                        original_quote = quote_pattern.search(original).group(0)
                        evidence_sources.append(EvidenceSource(
                            quote=original_quote, page_number=page["page_number"],
                            proposal_id=page.get("proposal_id"), section_type=page.get("section_type"),
                            speaker=page.get("speaker"),
                        ))
                    continue
                ev_raw = ev.strip()
                page_tag_match = re.search(r"\[(?:page|seite)\s+(\d+)\]", ev_raw, re.IGNORECASE)
                matched_page = int(page_tag_match.group(1)) if page_tag_match else None
                matched_prop = None

                ev_clean = re.sub(r"\[(?:page|seite)\s+\d+\]", "", ev_raw, flags=re.IGNORECASE)
                ev_clean = ev_clean.strip().strip('"').strip("«").strip("»").strip()

                if ev_clean:
                    for p in candidate_paras:
                        p_text = p.get("text", "")
                        if (
                            ev_clean.lower() in p_text.lower()
                            or (len(ev_clean) > 25 and p_text.lower() in ev_clean.lower())
                            or (len(ev_clean) > 30 and ev_clean[:40].lower() in p_text.lower())
                        ):
                            if matched_page is None:
                                matched_page = p.get("page_number")
                            matched_prop = p.get("proposal_id", 0)
                            break

                    if matched_page is None:
                        for p in all_paras:
                            p_text = p.get("text", "")
                            if (
                                ev_clean.lower() in p_text.lower()
                                or (len(ev_clean) > 25 and p_text.lower() in ev_clean.lower())
                                or (len(ev_clean) > 30 and ev_clean[:40].lower() in p_text.lower())
                            ):
                                matched_page = p.get("page_number")
                                matched_prop = p.get("proposal_id", 0)
                                break

                evidence_sources.append(
                    EvidenceSource(
                        quote=ev,
                        page_number=matched_page,
                        proposal_id=matched_prop,
                    )
                )
            final_evidence_list = nli_output.evidence

        if metrics["requested_strategy"] in ("hybrid", "hybrid_dense"):
            final_evidence_list = [source.quote for source in evidence_sources]
            if final_label == 0 and not evidence_sources:
                final_label, label_name = 1, config.LABEL_MAPPING[1]
                p_neutral, p_entail = max(0.85, p_neutral), 0.10
                decision_rule = "Vacuity Guardrail: No quote matches a selected original page -> Neutral (1)"

        return PredictionResult(
            id=case_id or "case-0001",
            claim=claim,
            label=final_label,
            label_name=label_name,
            reasoning=nli_output.reasoning,
            evidence=final_evidence_list,
            evidence_sources=evidence_sources,
            numerical_conflict=numerical_conflict_msg,
            p_entail=p_entail,
            p_neutral=p_neutral,
            p_contra=p_contra,
            fuzzy_rule=decision_rule,
            decision_rule=decision_rule,
            strategy=strat,
            booklet_path=str(booklet_pdf),
            tokens_prompt=nli_output.tokens_prompt,
            tokens_completion=nli_output.tokens_completion,
            tokens_total=nli_output.tokens_total,
            latency_ms=nli_output.latency_ms,
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
            retrieval_metrics=metrics,
            requested_strategy=metrics["requested_strategy"],
        )

    def verify_premise(
        self,
        claim: str,
        reference: str,
        claim_language: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> PredictionResult:
        """
        BEGINNER TASK: Direct NLI between claim (hypothesis) and reference string (premise).
        No PDF parsing required.
        """
        start_time = time.time()

        # 1. Deterministic Numerical Conflict Check
        num_conflict = detect_numerical_conflict(claim, reference)
        numerical_conflict_msg = num_conflict.explanation if num_conflict else None

        # 2. Apertus Model Inference over premise text
        nli_output = self.client.infer(
            context=reference,
            claim=claim,
            claim_language=claim_language,
        )

        final_label = nli_output.label
        decision_rule = nli_output.decision_rule_applied or nli_output.fuzzy_rule_applied
        p_entail = nli_output.p_entail
        p_contra = nli_output.p_contra
        p_neutral = nli_output.p_neutral

        if num_conflict and final_label != 2:
            final_label = 2
            p_contra = max(0.95, p_contra)
            decision_rule = f"Decision-Rule 1b: Numerical Clash ({num_conflict.claim_entity} vs {num_conflict.booklet_entity})"

        if final_label == 0:
            has_valid = any(len(ev.strip()) >= 15 for ev in nli_output.evidence) or len(reference.strip()) >= 15
            if not has_valid:
                final_label = 1
                p_neutral = max(0.85, p_neutral)
                p_entail = 0.10
                decision_rule = "Vacuity Guardrail: Ungrounded entailment prevented -> Neutral (1)"

        label_name = config.LABEL_MAPPING.get(final_label, "Unknown")

        # For Neutral (1), evidence is strictly []
        ev_sources: List[EvidenceSource] = []
        final_ev: List[str] = []

        if final_label != 1:
            clean_ref = re.sub(r"^\[(?:page|seite)\s+\d+\]\s*\d*\s*", "", reference, flags=re.IGNORECASE).strip(' "«»')
            ev_sources.append(EvidenceSource(quote=clean_ref, page_number=1, proposal_id=1))
            final_ev = [clean_ref]

        elapsed_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            id=case_id or "case-0001",
            claim=claim,
            label=final_label,
            label_name=label_name,
            reasoning=nli_output.reasoning,
            evidence=final_ev,
            evidence_sources=ev_sources,
            numerical_conflict=numerical_conflict_msg,
            p_entail=p_entail,
            p_neutral=p_neutral,
            p_contra=p_contra,
            fuzzy_rule=decision_rule,
            decision_rule=decision_rule,
            strategy="direct_reference",
            booklet_path="supplied_reference",
            tokens_prompt=nli_output.tokens_prompt,
            tokens_completion=nli_output.tokens_completion,
            tokens_total=nli_output.tokens_total,
            latency_ms=elapsed_ms,
        )
