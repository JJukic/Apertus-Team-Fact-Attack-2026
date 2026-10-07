"""
Apertus Client for Natural Language Inference.
Interacts with Apertus v1.5 (8B / 70B) using the OpenAI-compatible CSCS endpoint.
Measures token consumption and inference latency.
"""

import time
import json
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)


class ApertusError(RuntimeError):
    """Operational failure without an NLI label; usage is None when unknown."""

    def __init__(self, message, *, api_attempts=0, latency_ms=0.0,
                 tokens_prompt=None, tokens_completion=None, tokens_total=None):
        super().__init__(message)
        self.api_attempts = api_attempts
        self.latency_ms = latency_ms
        self.tokens_prompt = tokens_prompt
        self.tokens_completion = tokens_completion
        self.tokens_total = tokens_total


class ApertusConfigurationError(ApertusError):
    pass


class ApertusAPIError(ApertusError):
    pass


class ApertusResponseError(ApertusError):
    pass


class NLIOutput(BaseModel):
    label: int = Field(..., description="0 = Entailment, 1 = Neutral, 2 = Contradiction")
    reasoning: str = Field(..., description="Short explanation of the relationship")
    evidence: List[str] = Field(default_factory=list, description="Exact supporting passage(s) from document")
    p_entail: float = Field(default=0.0, description="Calibrated confidence for Entailment [0..1]")
    p_neutral: float = Field(default=0.0, description="Calibrated confidence for Neutral [0..1]")
    p_contra: float = Field(default=0.0, description="Calibrated confidence for Contradiction [0..1]")
    fuzzy_rule_applied: Optional[str] = Field(default=None, description="Applied decision rule (alias)")
    decision_rule_applied: Optional[str] = Field(default=None, description="Applied calibrated decision rule")
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    latency_ms: float = 0.0


class ApertusClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock: Optional[bool] = None,
    ):
        self.base_url = base_url or config.LLM_BASE_URL
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.model_name = model_name or config.LLM_NAME
        self.mock = mock if mock is not None else (config.MOCK_APERTUS or not bool(self.api_key))

        if not self.mock:
            if not self.api_key:
                raise ApertusConfigurationError("Live Apertus requires LLM_API_KEY")
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "EMPTY",
                    max_retries=0,
                )
            except Exception as e:
                logger.error("Could not initialize Apertus client (%s)", type(e).__name__)
                raise ApertusConfigurationError("Could not initialize Apertus client") from None
        else:
            self.client = None
            if not self.api_key:
                logger.info("No LLM_API_KEY provided; operating in local heuristic mock mode.")

    def infer(self, context: str, claim: str, claim_language: Optional[str] = None) -> NLIOutput:
        """
        Evaluate whether the booklet context entails, contradicts, or is neutral towards the claim.
        Returns a validated NLIOutput; raises ApertusError on operational failure.
        """
        start_time = time.time()

        if self.mock:
            # Deterministic heuristic mock for offline development and testing
            latency_ms = (time.time() - start_time) * 1000 + 15.0
            return self._mock_infer(context, claim, latency_ms)

        system_prompt = (
            "You are an expert multilingual document-grounded Natural Language Inference (NLI) system "
            "for official Swiss voting booklets (Abstimmungsbüchlein).\n"
            "Evaluate the logical relationship between the document context and the given claim.\n\n"
            "Classification Rules:\n"
            "- 0: Entailment -> The document context directly supports or confirms the claim.\n"
            "- 1: Neutral -> The document context does NOT mention the claim, or contains insufficient information.\n"
            "- 2: Contradiction -> The document context directly contradicts, refutes, or states the opposite of the claim.\n\n"
            "CRITICAL CHECKLIST:\n"
            "1. NUMBERS, QUANTITIES & DATES: If a claim cites a number, date, or percentage that differs from the document (e.g. 500'000 vs 1.7 million, or 2030 vs 2050), it is a DIRECT CONTRADICTION (2)!\n"
            "2. UNMENTIONED TOPICS: If a claim is about a topic not mentioned in the text, you MUST choose 1 (Neutral), never 2.\n"
            "3. CONTINUOUS CONFIDENCE: Estimate fuzzy membership degrees [0.0 to 1.0] for:\n"
            "   - p_entail: confidence of direct textual support\n"
            "   - p_neutral: confidence of missing/unaddressed information\n"
            "   - p_contra: confidence of factual, numerical, or logical contradiction\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "p_entail": <0.0 - 1.0>,\n'
            '  "p_neutral": <0.0 - 1.0>,\n'
            '  "p_contra": <0.0 - 1.0>,\n'
            '  "label": <0, 1, or 2>,\n'
            '  "reasoning": "<concise explanation in the claim language>",\n'
            '  "evidence": ["<verbatim supporting quote or passage from the document>"]\n'
            "}"
        )

        user_prompt = (
            f"=== DOCUMENT CONTEXT ===\n{context}\n\n"
            f"=== CLAIM ===\n{claim}\n\n"
            "Determine whether the document context entails (0), is neutral (1), or contradicts (2) the claim. "
            "Output JSON only:"
        )

        max_retries = 3
        response = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    timeout=45.0,
                )
                break
            except Exception as e:
                if getattr(e, "retryable", True) is False:
                    raise
                if attempt < max_retries - 1:
                    sleep_sec = (2 ** attempt) * 1.5
                    logger.warning("Apertus API attempt %s/%s failed (%s); retrying in %.1fs",
                                   attempt + 1, max_retries, type(e).__name__, sleep_sec)
                    time.sleep(sleep_sec)
                else:
                    logger.error("Apertus API failed after %s attempts (%s)",
                                 max_retries, type(e).__name__)

        if response is None:
            raise ApertusAPIError(
                "Apertus API failed after 3 attempts", api_attempts=max_retries,
                latency_ms=round((time.time() - start_time) * 1000, 2)) from None

        usage = getattr(response, "usage", None)
        measured_usage = {name: getattr(usage, field, None) for name, field in (
            ("tokens_prompt", "prompt_tokens"), ("tokens_completion", "completion_tokens"),
            ("tokens_total", "total_tokens"))}
        try:
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) in ("length", "content_filter"):
                raise ValueError("Incomplete model response")
            content = choice.message.content.strip()
            parsed_json = self._parse_json(content)
            raw_label = parsed_json["label"]
            if type(raw_label) is not int or raw_label not in (0, 1, 2):
                raise ValueError("Invalid NLI label")
            probabilities = [parsed_json[name] for name in ("p_entail", "p_neutral", "p_contra")]
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities):
                raise ValueError("Invalid confidence value")
            p_entail, p_neutral, p_contra = probabilities
            final_label, decision_rule = self._apply_calibrated_decision(
                p_entail, p_neutral, p_contra, raw_label)
            evidence = parsed_json["evidence"]
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
                raise ValueError("Invalid evidence list")
            if not isinstance(parsed_json["reasoning"], str):
                raise ValueError("Invalid reasoning")
            tokens_prompt = measured_usage["tokens_prompt"]
            if tokens_prompt is None:
                tokens_prompt = self._estimate_tokens(user_prompt)
            tokens_completion = measured_usage["tokens_completion"]
            if tokens_completion is None:
                tokens_completion = self._estimate_tokens(content)
            tokens_total = measured_usage["tokens_total"]
            if tokens_total is None:
                tokens_total = tokens_prompt + tokens_completion
            return NLIOutput(
                label=final_label, reasoning=parsed_json["reasoning"], evidence=evidence,
                p_entail=p_entail, p_neutral=p_neutral, p_contra=p_contra,
                fuzzy_rule_applied=decision_rule, decision_rule_applied=decision_rule,
                tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                latency_ms=round((time.time() - start_time) * 1000, 2))
        except Exception as e:
            logger.error("Invalid Apertus response (%s)", type(e).__name__)
            raise ApertusResponseError(
                "Invalid or incomplete Apertus response", api_attempts=attempt + 1,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                **measured_usage) from None

    @classmethod
    def _apply_fuzzy_decision(cls, p_entail: float, p_neutral: float, p_contra: float, raw_label: int) -> Tuple[int, Optional[str]]:
        """Backward-compatible alias for _apply_calibrated_decision."""
        return cls._apply_calibrated_decision(p_entail, p_neutral, p_contra, raw_label)

    @staticmethod
    def _apply_calibrated_decision(p_entail: float, p_neutral: float, p_contra: float, raw_label: int) -> Tuple[int, Optional[str]]:
        """
        Calibrated decision arbiter:
        Applies empirical threshold rules over the model's confidence distribution
        [p_entail, p_neutral, p_contra] to resolve epistemic ambiguity and numerical conflicts.
        """
        total_p = p_entail + p_neutral + p_contra
        if total_p == 0.0:
            return raw_label, None

        # Rule 1: Strong contradiction or numerical conflict signal
        if p_contra >= 0.40 and p_contra > p_entail:
            return 2, "Decision-Rule 1: Conflict dominant (p_contra >= 0.40)"

        # Rule 2: Strong direct support
        if p_entail >= 0.60 and p_contra < 0.25:
            return 0, "Decision-Rule 2: Direct support dominant (p_entail >= 0.60)"

        # Rule 3: High neutral probability or genuine ambiguity between entail and contra
        if p_neutral >= 0.40:
            return 1, "Decision-Rule 3: High neutrality (p_neutral >= 0.40)"
        if (p_entail > 0.0 or p_contra > 0.0) and abs(p_entail - p_contra) < 0.15:
            return 1, "Decision-Rule 3: Epistemic ambiguity zone (|p_entail - p_contra| < 0.15)"

        return raw_label, None

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        def parse(candidate):
            def reject_constant(value):
                raise ValueError("Non-finite JSON number")
            parsed = json.loads(candidate, strict=False, parse_constant=reject_constant)
            if not isinstance(parsed, dict):
                raise ValueError("JSON response must be an object")
            return parsed

        try:
            return parse(text)
        except ValueError:
            # Accept complete JSON objects wrapped in explanatory text.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return parse(text[start:end + 1])
                except ValueError:
                    pass
            raise ValueError("Could not parse model response as a JSON object") from None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Multilingual token estimation for Swiss languages (DE, FR, IT).
        Empirically calibrated to Apertus/Llama tokenizers: ~3.7 characters per token.
        """
        if not text:
            return 0
        return max(1, int(len(text) / 3.7))

    def _mock_infer(self, context: str, claim: str, latency_ms: float) -> NLIOutput:
        """
        Mock inference for offline testing.
        Uses lexical overlap heuristics to generate a plausible response.
        """
        prompt_est = self._estimate_tokens(context) + self._estimate_tokens(claim) + 50
        comp_est = 45

        # Heuristic: check if claim words appear in context
        claim_words = [w.lower() for w in claim.split() if len(w) > 4]
        match_count = sum(1 for w in claim_words if w in context.lower())

        if not claim_words or match_count == 0:
            label = 1  # Neutral
            reasoning = "Das offizielle Abstimmungsbüchlein enthält zu dieser Fragestellung keine definitive Aussage."
            evidence = []
        elif "nicht" in claim.lower() or "kein" in claim.lower() or "500'000" in claim:
            label = 2  # Contradiction
            reasoning = "Die Behauptung steht im Widerspruch zu den offiziellen Zahlen oder Empfehlungen im Abstimmungsbüchlein."
            evidence = [context[:200]]
        else:
            label = 0  # Entailment
            reasoning = "Die Kernaussage wird durch die amtlichen Ausführungen des Abstimmungsbüchleins gestützt und bestätigt."
            evidence = [context[:200]]

        return NLIOutput(
            label=label,
            reasoning=reasoning,
            evidence=evidence,
            tokens_prompt=prompt_est,
            tokens_completion=comp_est,
            tokens_total=prompt_est + comp_est,
            latency_ms=round(latency_ms, 2),
        )
