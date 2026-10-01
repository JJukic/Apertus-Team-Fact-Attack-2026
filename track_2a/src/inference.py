"""
Inference Pipeline for Claim Verification over Swiss Voting Booklets.
Coordinates PDF parsing, passage retrieval, and Apertus LLM inference.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient, NLIOutput
from src import config


class PredictionResult(BaseModel):
    claim: str
    label: int
    label_name: str
    reasoning: str
    evidence: List[str]
    p_entail: float = 0.0
    p_neutral: float = 0.0
    p_contra: float = 0.0
    fuzzy_rule: Optional[str] = None
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
        """
        strat = strategy or self.strategy
        booklet_data = self._get_booklet_data(booklet_pdf)

        if strat == "full":
            # Strategy 1: Provide full booklet text to Apertus
            context = booklet_data["full_text"]
        else:
            # Strategy 2: Retrieve top-k relevant paragraphs
            retriever: PassageRetriever = booklet_data["retriever"]
            relevant_paras = retriever.retrieve(claim, top_k=top_k)
            context_blocks = []
            for p in relevant_paras:
                context_blocks.append(f"[Page {p['page_number']}] {p['text']}")
            context = "\n\n".join(context_blocks)

        nli_output: NLIOutput = self.client.infer(
            context=context,
            claim=claim,
            claim_language=claim_language,
        )

        label_name = config.LABEL_MAPPING.get(nli_output.label, "Unknown")

        return PredictionResult(
            claim=claim,
            label=nli_output.label,
            label_name=label_name,
            reasoning=nli_output.reasoning,
            evidence=nli_output.evidence,
            p_entail=nli_output.p_entail,
            p_neutral=nli_output.p_neutral,
            p_contra=nli_output.p_contra,
            fuzzy_rule=nli_output.fuzzy_rule_applied,
            strategy=strat,
            booklet_path=str(booklet_pdf),
            tokens_prompt=nli_output.tokens_prompt,
            tokens_completion=nli_output.tokens_completion,
            tokens_total=nli_output.tokens_total,
            latency_ms=nli_output.latency_ms,
        )
