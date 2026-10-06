"""
Apertus Client for Natural Language Inference.
Interacts with Apertus v1.5 (8B / 70B) using the OpenAI-compatible CSCS endpoint.
Measures token consumption and inference latency.
"""

import time
import json
import logging
import math
import random
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)


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
    error: Optional[str] = None
    extracted_statements: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[int] = Field(default_factory=list, description="1-based ids of cited passages (compact mode)")


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
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "EMPTY",
                    max_retries=0,  # retries are handled by _create_with_retry
                )
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Falling back to mock mode.")
                self.mock = True
        else:
            self.client = None
            if not self.api_key:
                logger.info("No LLM_API_KEY provided; operating in local heuristic mock mode.")

    COMPACT_SYSTEM_PROMPT = (
        "You are a natural language inference (NLI) judge for official Swiss federal voting booklets.\n"
        "The passages and the claim may be in different languages (German, French, Italian): compare meaning, not wording.\n"
        "Decide how the PASSAGES relate to the CLAIM:\n"
        "0 = entailment: the passages support the claim.\n"
        "1 = neutral: the passages do not provide enough information to confirm or refute the claim "
        "(e.g. the detail, number, date or topic is not mentioned).\n"
        "2 = contradiction: the passages state something incompatible with the claim "
        "(e.g. a different number or date, the opposite recommendation, a negated statement).\n"
        "Judge only against the passages, never against world knowledge.\n"
        "Important: if the passages are about a different topic or simply do not address the specific statement of the claim, "
        "the answer is 1 (neutral), even if the claim sounds false. Choose 2 only when a passage explicitly states the opposite "
        "or a conflicting fact about the same subject.\n"
        "Answer format: the label digit, then '|', then the ids of the passages that justify the label, "
        "e.g. '0|P2' or '2|P3,P7'. For neutral answer '1|'. No other text."
    )

    def infer_compact(
        self,
        passages: List[str],
        claim: str,
        claim_language: Optional[str] = None,
        vote: Optional[str] = None,
    ) -> NLIOutput:
        """
        Token-efficient NLI: passages are numbered [P1]..[Pn]; the model answers '<label>|<ids>'.
        Label probabilities come from the logprobs of the first output token, and evidence is
        returned as passage ids, so quotes are always verbatim booklet text.
        """
        start_time = time.time()
        numbered = "\n\n".join(f"[P{i}] {txt}" for i, txt in enumerate(passages, 1))
        lang = f" ({claim_language})" if claim_language else ""
        vote_line = f"VOTE: {vote}\n" if vote else ""
        user_prompt = f"PASSAGES:\n{numbered}\n\n{vote_line}CLAIM{lang}: {claim}\nAnswer:"

        if self.mock:
            out = self._mock_infer("\n".join(passages), claim, 15.0)
            out.evidence_ids = [1] if out.label != 1 and passages else []
            return out

        response, err = self._chat(
            [{"role": "system", "content": self.COMPACT_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_tokens=24,
            logprobs=True,
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        if response is None:
            return NLIOutput(label=1, reasoning=f"API Error: {err}", error=f"api: {err}", latency_ms=latency_ms)

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        m = re.fullmatch(r"([012])\|\s*((?:P?\d+\s*(?:,\s*P?\d+\s*)*)?)", content)
        probs = self._label_probs(choice)
        if not m:
            return NLIOutput(label=1, reasoning="Invalid compact response", error="parse: invalid compact response", latency_ms=latency_ms)
        label = int(m.group(1))
        ids = list(dict.fromkeys(int(tok) for tok in re.findall(r"\d+", m.group(2))))
        if any(not 1 <= i <= len(passages) for i in ids):
            return NLIOutput(label=1, reasoning="Invalid passage id", error="parse: invalid passage id", latency_ms=latency_ms)

        usage = getattr(response, "usage", None)
        tp = getattr(usage, "prompt_tokens", None) or self._estimate_tokens(self.COMPACT_SYSTEM_PROMPT + user_prompt)
        tc = getattr(usage, "completion_tokens", None) or self._estimate_tokens(content)
        return NLIOutput(
            label=label,
            reasoning=f"Apertus answer: {content}",
            evidence=[passages[i - 1] for i in ids],
            evidence_ids=ids,
            p_entail=probs.get(0, 0.0),
            p_neutral=probs.get(1, 0.0),
            p_contra=probs.get(2, 0.0),
            decision_rule="logprob argmax" if probs else None,
            tokens_prompt=tp,
            tokens_completion=tc,
            tokens_total=tp + tc,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _label_probs(choice) -> Dict[int, float]:
        """Normalised P(label) from the top logprobs of the first label-like output token."""
        lp = getattr(choice, "logprobs", None)
        if not lp or not lp.content:
            return {}
        for tok in lp.content[:3]:
            cands = {int(t.token.strip()): math.exp(t.logprob) for t in (tok.top_logprobs or []) if t.token.strip() in ("0", "1", "2")}
            if cands:
                total = sum(cands.values())
                return {k: round(v / total, 4) for k, v in cands.items()}
        return {}

    # Worked examples for the main error class: claims attributed to one side of the booklet.
    # Text from the 2026-06-14 booklet; the claims are written for this prompt (not taken from the dataset).
    FEW_SHOT_EXAMPLES = (
        "\n\nEXAMPLES (same passages for all three claims):\n"
        "[P1] (p. 16; Arguments of the initiative/referendum committee) Die ständige Wohnbevölkerung hat innerhalb "
        "von 12 Jahren um 1 Million Menschen zugenommen. Der Hauptgrund ist die massive Zuwanderung. "
        "Die Mieten werden immer teurer.\n"
        "[P2] (p. 18; Arguments of the Federal Council and Parliament) Die Initiative schadet dem Wohlstand. "
        "Schweizer Unternehmen sind auf ausländische Arbeitskräfte angewiesen. Die Wirtschaft und der Wohlstand "
        "der Schweiz würden leiden.\n"
        "CLAIM (fr): Selon le comité, l'immigration massive est la principale raison de la croissance démographique.\n"
        '{"p_entail": 1.0, "p_neutral": 0.0, "p_contra": 0.0, "label": 0, "evidence_ids": [1]}\n'
        "(The committee says this in P1. P2 argues against the initiative but does not contradict the committee's statement.)\n"
        "CLAIM (it): Secondo il comitato, la popolazione residente è diminuita negli ultimi dodici anni.\n"
        '{"p_entail": 0.0, "p_neutral": 0.0, "p_contra": 1.0, "label": 2, "evidence_ids": [1]}\n'
        "(The committee says the population grew by 1 million.)\n"
        "CLAIM (de): Der Bundesrat erwartet, dass die Initiative die Mieten in den Städten senkt.\n"
        '{"p_entail": 0.0, "p_neutral": 1.0, "p_contra": 0.0, "label": 1, "evidence_ids": []}\n'
        "(No passage says what the Federal Council expects for rents.)\n"
    )

    LANGUAGE_NAMES = {"de": "German", "fr": "French", "it": "Italian"}

    def translate(self, text: str, target_lang: str) -> Tuple[Optional[str], int, int, float]:
        """Translate a claim into the booklet language. Returns (translation or None, prompt tokens, completion tokens, ms)."""
        start = time.time()
        if self.mock:
            return None, 0, 0, 0.0
        target = self.LANGUAGE_NAMES.get(target_lang, target_lang)
        response, _ = self._chat([
            {"role": "system", "content": f"Translate the user's sentence into {target}. Keep names, numbers and "
                                          "who says what exactly. Output only the translation."},
            {"role": "user", "content": text},
        ], max_tokens=200)
        ms = (time.time() - start) * 1000
        if response is None:
            return None, 0, 0, ms
        out = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        return (out or None, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0, ms)

    @staticmethod
    def _retry_limit(error: Exception) -> int:
        """
        How many retries an API error deserves. Timeouts, dropped connections, 429 and 5xx are transient.
        Under load the CSCS gateway also answers 'invalid API key' to valid keys (seen interleaved with successful
        calls; 403 likewise), so 401/403 get two quick retries, but not the full budget, so a really wrong key fails fast.
        Other client errors (400 bad request, 404 unknown model) are not retried.
        """
        status = getattr(error, "status_code", None)
        if status is None or status in (408, 409, 425, 429) or status >= 500:
            return config.LLM_MAX_RETRIES
        if status in (401, 403):
            return min(2, config.LLM_MAX_RETRIES)
        return 0

    def _create_with_retry(self, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """chat.completions.create with exponential backoff (+ jitter) inside a per-request time budget."""
        start = time.time()
        attempt = 0
        while True:
            try:
                return self.client.chat.completions.create(**kwargs), None
            except Exception as e:
                delay = min(30.0, 2.0 * 2 ** attempt) * random.uniform(0.5, 1.0)
                if attempt >= self._retry_limit(e) or time.time() - start + delay > config.LLM_RETRY_BUDGET_S:
                    logger.error(f"Apertus API query failed after {attempt + 1} attempt(s): {e}")
                    return None, e
                attempt += 1
                logger.warning(f"Apertus API attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    def _chat(self, messages: List[Dict[str, str]], max_tokens: int, logprobs: bool = False):
        kwargs = dict(model=self.model_name, messages=messages, temperature=0.0, max_tokens=max_tokens, timeout=60.0)
        if logprobs:
            kwargs.update(logprobs=True, top_logprobs=5)
        return self._create_with_retry(**kwargs)

    def infer(
        self,
        context: str,
        claim: str,
        claim_language: Optional[str] = None,
        passages: Optional[List[str]] = None,
    ) -> NLIOutput:
        """
        Evaluate whether the booklet context entails, contradicts, or is neutral towards the claim.
        Returns NLIOutput with label (0, 1, 2), reasoning, evidence, tokens, and latency.

        If `passages` is given ('ids' mode), they are numbered [P1]..[Pn] and replace `context`;
        the model cites passage ids instead of writing quotes and reasoning (far fewer output tokens).
        """
        start_time = time.time()
        if passages is not None:
            context = "\n\n".join(f"[P{i}] {txt}" for i, txt in enumerate(passages, 1))

        if self.mock:
            # Deterministic heuristic mock for offline development and testing
            latency_ms = (time.time() - start_time) * 1000 + 15.0
            out = self._mock_infer(context, claim, latency_ms)
            if passages is not None and out.label != 1 and passages:
                out.evidence_ids = [1]
                out.evidence = [passages[0]]
            return out

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
            "2b. VIEWPOINTS: The booklet deliberately contains opposing viewpoints (Federal Council and Parliament vs. the "
            "initiative or referendum committee). If the claim attributes a statement to an actor (e.g. 'the committee argues...', "
            "'the Federal Council recommends...', 'according to the summary...'), check only whether THAT actor's text supports it. "
            "Arguments of the other side are NOT a contradiction.\n"
            "2c. RECOMMENDATIONS: 'recommends rejecting X' is supported (0) by text where the Federal Council/Parliament recommend "
            "'No' on X or reject X; it is contradicted (2) only if they recommend 'Yes'.\n"
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
        if passages is not None and config.FEW_SHOT:
            system_prompt = system_prompt + self.FEW_SHOT_EXAMPLES
        if passages is not None and config.IDS_REASON:
            system_prompt = system_prompt.replace(
                '{\n  "p_entail"',
                '{\n  "reason": "<max. 15 words: what the most relevant passage says about the claim>",\n  "p_entail"',
            )
        if passages is not None:
            system_prompt = system_prompt.replace(
                '  "reasoning": "<concise explanation in the claim language>",\n'
                '  "evidence": ["<verbatim supporting quote or passage from the document>"]\n',
                '  "evidence_ids": [<ids of the passages that justify the label, e.g. 2, 5; empty for 1>]\n',
            ).replace(
                "Evaluate the logical relationship between the document context and the given claim.\n",
                "Evaluate the logical relationship between the document context and the given claim. "
                "The context is a list of numbered passages [P1], [P2], ... and may be in a different "
                "language (German, French, Italian) than the claim: compare meaning, not wording.\n",
            )

        thinking = passages is not None and config.THINKING
        if thinking:
            system_prompt = system_prompt.replace(
                "Respond ONLY with a valid JSON object matching this schema:\n",
                "First think briefly (at most 120 words): who (if anyone) the claim attributes the statement to, "
                "which passages belong to that speaker, and what exactly they say about the claim. Then give your "
                "final answer as a valid JSON object matching this schema:\n",
            )

        user_prompt = (
            f"=== DOCUMENT CONTEXT ===\n{context}\n\n"
            f"=== CLAIM ===\n{claim}\n\n"
            "Determine whether the document context entails (0), is neutral (1), or contradicts (2) the claim. "
            + ("Think it through, then output the JSON:" if thinking else "Output JSON only:")
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra: Dict[str, Any] = {}
        if thinking:
            # Asked to think, Apertus still answers with the JSON first and explains afterwards; prefilling the
            # reasoning marker makes it reason before deciding
            messages.append({"role": "assistant", "content": "<|inner_prefix|>"})
            extra["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}

        budget = config.THINKING_BUDGET if thinking else 0
        response, last_exception = self._create_with_retry(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=(budget or config.THINKING_MAX_TOKENS) if thinking else (512 if passages is None else 160),
            timeout=180.0 if thinking else 45.0,
            **extra,
        )

        # Thinking budget used up before a decision: close the reasoning and let the model write the answer
        forced_prompt = forced_completion = 0
        if response is not None and budget:
            partial = (response.choices[0].message.content or "").rstrip()
            if "<|inner_suffix|>" not in partial:
                answer, _ = self._create_with_retry(
                    model=self.model_name,
                    messages=messages[:-1] + [{"role": "assistant", "content": f"<|inner_prefix|>{partial} …<|inner_suffix|>"}],
                    temperature=0.0,
                    max_tokens=160,
                    timeout=45.0,
                    **extra,
                )
                if answer is not None:
                    response.choices[0].message.content = f"{partial} …<|inner_suffix|>{answer.choices[0].message.content or ''}"
                    usage = getattr(answer, "usage", None)
                    forced_prompt = getattr(usage, "prompt_tokens", 0) or 0
                    forced_completion = getattr(usage, "completion_tokens", 0) or 0

        if response is None:
            latency_ms = (time.time() - start_time) * 1000
            return NLIOutput(
                label=1,
                reasoning=f"API Error: {str(last_exception)}",
                evidence=[],
                error=f"api: {last_exception}",
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                latency_ms=round(latency_ms, 2),
            )

        try:
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content.strip()
            parsed_json = self._parse_json(self._strip_thinking(content, thinking=thinking))

            usage = getattr(response, "usage", None)
            tokens_prompt = usage.prompt_tokens if (usage and getattr(usage, "prompt_tokens", None)) else self._estimate_tokens(user_prompt)
            tokens_completion = usage.completion_tokens if (usage and getattr(usage, "completion_tokens", None)) else self._estimate_tokens(content)
            tokens_total = usage.total_tokens if (usage and getattr(usage, "total_tokens", None)) else (tokens_prompt + tokens_completion)
            tokens_prompt += forced_prompt
            tokens_completion += forced_completion
            tokens_total += forced_prompt + forced_completion

            label_given = "label" in parsed_json
            raw_label = parsed_json.get("label")
            if type(raw_label) is not int or raw_label not in (0, 1, 2):
                raise ValueError("label must be an integer in 0, 1, 2")

            p_entail = float(parsed_json.get("p_entail", 0.0))
            p_neutral = float(parsed_json.get("p_neutral", 0.0))
            p_contra = float(parsed_json.get("p_contra", 0.0))

            if not all(math.isfinite(p) and 0 <= p <= 1 for p in (p_entail, p_neutral, p_contra)):
                raise ValueError("confidences must be finite values between 0 and 1")

            # Apply Calibrated Decision Arbiter
            final_label, decision_rule = self._apply_calibrated_decision(
                p_entail, p_neutral, p_contra, raw_label, label_given=label_given)

            evidence = parsed_json.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]

            evidence_ids: List[int] = []
            if passages is not None:
                raw_ids = parsed_json.get("evidence_ids", [])
                if not isinstance(raw_ids, list) or any(type(i) is not int or not 1 <= i <= len(passages) for i in raw_ids):
                    raise ValueError("evidence_ids must contain valid integer passage ids")
                evidence_ids = list(dict.fromkeys(raw_ids))
                evidence = [passages[i - 1] for i in evidence_ids]

            return NLIOutput(
                label=final_label,
                reasoning=parsed_json.get("reasoning", "") or f"Apertus answer: {content}",
                evidence=evidence,
                evidence_ids=evidence_ids,
                p_entail=p_entail,
                p_neutral=p_neutral,
                p_contra=p_contra,
                fuzzy_rule_applied=decision_rule,
                decision_rule_applied=decision_rule,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as e:
            logger.error(f"Error parsing Apertus response: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return NLIOutput(
                label=1,
                reasoning=f"Response Parse Error: {str(e)}",
                evidence=[],
                error=f"parse: {e}",
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                latency_ms=round(latency_ms, 2),
            )

    @classmethod
    def _apply_fuzzy_decision(cls, p_entail: float, p_neutral: float, p_contra: float, raw_label: int) -> Tuple[int, Optional[str]]:
        """Backward-compatible alias for _apply_calibrated_decision."""
        return cls._apply_calibrated_decision(p_entail, p_neutral, p_contra, raw_label)

    @staticmethod
    def _apply_calibrated_decision(
        p_entail: float, p_neutral: float, p_contra: float, raw_label: int, label_given: bool = False,
    ) -> Tuple[int, Optional[str]]:
        """
        Calibrated decision arbiter:
        Applies empirical threshold rules over the model's confidence distribution
        [p_entail, p_neutral, p_contra] to resolve epistemic ambiguity and numerical conflicts.
        """
        total_p = p_entail + p_neutral + p_contra
        if total_p == 0.0:
            return raw_label, None

        # Rule 0: a self-inconsistent answer ("label": 0 with "p_contra": 1.0) keeps the model's label.
        # Over all saved ids-mode runs the label was right in 44 of these cases and the probabilities in 13.
        probs = (p_entail, p_neutral, p_contra)
        if label_given and probs[raw_label] < max(probs):
            # Rule 0b: except "label": 1 (Neutral) with the confidence on Entailment/Contradiction. That answer was
            # right in 1 of 18 such cases over all 1,495 OST pairs; the confidences' choice fixed 8 and broke 1
            if raw_label == 1:
                return probs.index(max(probs)), "Decision-Rule 0b: Neutral label but confidences name a relation -> confidences"
            return raw_label, "Decision-Rule 0: label and confidences disagree -> model label kept"

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
    def _strip_thinking(text: str, thinking: bool = False) -> str:
        """
        The thinking model writes '<|inner_prefix|>reasoning<|inner_suffix|>answer'. Return only the answer;
        if the reasoning was cut off before the suffix, fall back to the last JSON object that has a label.
        """
        if "<|inner_suffix|>" in text:
            return text.rsplit("<|inner_suffix|>", 1)[1].strip()
        if thinking or "<|inner_prefix|>" in text:
            objects = re.findall(r"\{[^{}]*\"label\"[^{}]*\}", text)
            return objects[-1] if objects else text
        return text

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
        try:
            return json.loads(text, strict=False)
        except Exception:
            # Fallback search for JSON object inside braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1], strict=False)
                except Exception:
                    pass
            
            raise ValueError("Could not parse model response as JSON")

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
