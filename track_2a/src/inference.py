"""
Inference Pipeline for Claim Verification over Swiss Voting Booklets.
Coordinates PDF parsing, passage retrieval, numerical conflict detection,
page-attributed evidence extraction, and Apertus LLM inference.
"""

import re
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient, NLIOutput
from src.numerical_checker import detect_numerical_conflict, NumericalConflictResult
from src import config


class EvidenceSource(BaseModel):
    quote: str
    page_number: Optional[int] = None
    proposal_id: Optional[int] = None


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
    ):
        self.strategy = strategy
        self.client = apertus_client or ApertusClient()
        self.pdf_parser = PDFParser()

    def _get_booklet_data(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        path_str = str(Path(pdf_path).resolve())
        if path_str not in _GLOBAL_BOOKLET_CACHE:
            pages = self.pdf_parser.extract_pages(path_str)
            paragraphs = self.pdf_parser.extract_paragraphs(path_str)
            full_text = self.pdf_parser.extract_full_text(path_str)
            retriever = PassageRetriever(paragraphs)
            _GLOBAL_BOOKLET_CACHE[path_str] = {
                "pages": pages,
                "paragraphs": paragraphs,
                "full_text": full_text,
                "retriever": retriever,
            }
        return _GLOBAL_BOOKLET_CACHE[path_str]

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
        strat = strategy or self.strategy
        booklet_data = self._get_booklet_data(booklet_pdf)

        candidate_paras = []
        if strat == "full":
            # Strategy 1: Provide full booklet text to Apertus
            context = booklet_data["full_text"]
            candidate_paras = booklet_data["paragraphs"]
        else:
            # Strategy 2: Retrieve top-k relevant paragraphs (with proposal isolation & vote targeting)
            retriever: PassageRetriever = booklet_data["retriever"]
            candidate_paras = retriever.retrieve(claim, top_k=top_k, target_vote=vote)
            context_blocks = []
            for p in candidate_paras:
                context_blocks.append(f"[Page {p['page_number']}] {p['text']}")
            context = "\n\n".join(context_blocks)

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
