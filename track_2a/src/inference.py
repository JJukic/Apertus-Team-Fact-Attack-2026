"""
Inference Pipeline for Claim Verification over Swiss Voting Booklets.
Coordinates PDF parsing, passage retrieval, numerical conflict detection,
page-attributed evidence extraction, and Apertus LLM inference.
"""

import re
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


class ClaimVerificationEngine:
    def __init__(
        self,
        strategy: str = config.DEFAULT_STRATEGY,
        apertus_client: Optional[ApertusClient] = None,
    ):
        self.strategy = strategy
        self.client = apertus_client or ApertusClient()
        self.pdf_parser = PDFParser()
        self._booklet_cache: Dict[str, Dict[str, Any]] = {}

    def _get_booklet_data(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        path_str = str(Path(pdf_path).resolve())
        if path_str not in self._booklet_cache:
            paragraphs = self.pdf_parser.extract_paragraphs(path_str)
            full_text = self.pdf_parser.extract_full_text(path_str)
            retriever = PassageRetriever(paragraphs)
            self._booklet_cache[path_str] = {
                "paragraphs": paragraphs,
                "full_text": full_text,
                "retriever": retriever,
            }
        return self._booklet_cache[path_str]

    def verify_claim(
        self,
        claim: str,
        booklet_pdf: Union[str, Path],
        claim_language: Optional[str] = None,
        strategy: Optional[str] = None,
        top_k: int = 5,
    ) -> PredictionResult:
        """
        Verify whether the voting booklet entails, contradicts, or is neutral to the claim.
        Incorporates deterministic numerical conflict checks and page-attributed evidence.
        """
        strat = strategy or self.strategy
        booklet_data = self._get_booklet_data(booklet_pdf)

        candidate_paras = []
        if strat == "full":
            # Strategy 1: Provide full booklet text to Apertus
            context = booklet_data["full_text"]
            candidate_paras = booklet_data["paragraphs"]
        else:
            # Strategy 2: Retrieve top-k relevant paragraphs
            retriever: PassageRetriever = booklet_data["retriever"]
            candidate_paras = retriever.retrieve(claim, top_k=top_k)
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

        label_name = config.LABEL_MAPPING.get(final_label, "Unknown")

        # 3. Match evidence quotes to exact source page numbers & proposals
        evidence_sources: List[EvidenceSource] = []
        all_paras = booklet_data["paragraphs"]

        for ev in nli_output.evidence:
            ev_raw = ev.strip()
            # 1. Extract [Page X] / [Seite X] tag if present in the model or mock quote
            page_tag_match = re.search(r"\[(?:page|seite)\s+(\d+)\]", ev_raw, re.IGNORECASE)
            matched_page = int(page_tag_match.group(1)) if page_tag_match else None
            matched_prop = None

            # Clean tags, brackets, and quotes
            ev_clean = re.sub(r"\[(?:page|seite)\s+\d+\]", "", ev_raw, flags=re.IGNORECASE)
            ev_clean = ev_clean.strip().strip('"').strip("«").strip("»").strip()

            if ev_clean:
                # First check candidate paragraphs
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

                # Fallback check all paragraphs in booklet
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

        return PredictionResult(
            claim=claim,
            label=final_label,
            label_name=label_name,
            reasoning=nli_output.reasoning,
            evidence=nli_output.evidence,
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
