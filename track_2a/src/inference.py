"""
Inference Pipeline for Claim Verification over Swiss Voting Booklets.
Coordinates PDF parsing, passage retrieval, numerical conflict detection,
page-attributed evidence extraction, and Apertus LLM inference.
"""

import hashlib
import json
import re
import time
import threading
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel, Field, StrictInt

from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.apertus_client import ApertusClient, NLIOutput
from src.numerical_checker import detect_numerical_conflict, NumericalConflictResult
from src import config
from src.text_utils import clip_to_query, guess_language, official_normalize


class EvidenceSource(BaseModel):
    quote: str
    page_number: Optional[StrictInt] = None
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
    error: Optional[str] = None
    context_pages: List[int] = Field(default_factory=list)  # booklet pages supplied to Apertus
    retrieval_query_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_official_dict(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export output strictly conforming to the Hack Apertus Track 2A (OST) JSON Schema.
        For Neutral (1), evidence is strictly an empty list [].
        """
        cid = case_id if case_id is not None else (self.id if self.id is not None else "case-0001")
        ev_list: List[Dict[str, Any]] = []
        if self.label not in config.LABEL_MAPPING:
            raise ValueError("Invalid NLI label")
        beginner = self.strategy == "direct_reference"
        seen = set()
        if self.label != 1:
            for source in self.evidence_sources:
                page = None if beginner else source.page_number
                if not beginner and (type(page) is not int or page < 1):
                    continue  # An unknown physical page is never attributed to page 1.
                # Only remove the engine's synthetic prefix; retain source numbers,
                # quotation marks, Unicode and original whitespace verbatim.
                quote = re.sub(r"^\[(?:page|seite)\s+\d+\] ?", "", source.quote, flags=re.IGNORECASE)
                normalized = official_normalize(quote)
                identity = (page, normalized)
                if normalized and len(normalized) <= 5000 and identity not in seen:
                    ev_list.append({"page": page, "text": quote})
                    seen.add(identity)
                if len(ev_list) == 5:
                    break
        if config.OFFICIAL_IO and not beginner and self.label != 1 and not ev_list:
            raise ValueError("Non-neutral Advanced prediction has no attributed evidence")

        return {
            "id": cid,
            "label": self.label,
            "label_name": config.LABEL_MAPPING[self.label].lower(),
            "evidence": ev_list,
            "metrics": {
                "input_tokens": int(self.tokens_prompt),
                "output_tokens": int(self.tokens_completion),
                "inference_time_ms": int(round(self.latency_ms)),
            },
        }


_GLOBAL_BOOKLET_CACHE: Dict[str, Dict[str, Any]] = {}
_BOOKLET_LOCKS: Dict[Any, threading.Lock] = {}
_BOOKLET_LOCK_GUARD = threading.Lock()
_PARSE_CACHE_VERSION = 6  # bump when parsing or section detection changes

_COMMITTEE_SECTION = "Arguments of the initiative/referendum committee"
_FEDERAL_COUNCIL_SECTION = "Arguments of the Federal Council and Parliament"
_COMMITTEE_RE = re.compile(
    r"komitee|initiant|urheber|comit[eé]|auteurs de l.initiative|comitato|promotori", re.IGNORECASE)
_FEDERAL_COUNCIL_RE = re.compile(
    r"bundesrat|parlament|conseil f[eé]d[eé]ral|consiglio federale|parlamento", re.IGNORECASE)


def opposing_sections(claim: str) -> Optional[set]:
    """
    Speaker-aware retrieval: if a claim attributes a statement to one side, hide the other side's argument pages.
    Apertus otherwise cites the Federal Council's counter-arguments as a 'contradiction' of a committee claim,
    even when told not to (committee-attributed claims had a 19 % error rate vs. 5-7 % for all others on dev).
    """
    committee = bool(_COMMITTEE_RE.search(claim))
    federal_council = bool(_FEDERAL_COUNCIL_RE.search(claim))
    if committee and not federal_council:
        return {_FEDERAL_COUNCIL_SECTION}
    if federal_council and not committee:
        return {_COMMITTEE_SECTION}
    return None


def attributed_section(claim: str) -> Optional[str]:
    """The argument section of the side a claim attributes its statement to (committee or Federal Council), if exactly one."""
    committee = bool(_COMMITTEE_RE.search(claim))
    federal_council = bool(_FEDERAL_COUNCIL_RE.search(claim))
    if committee != federal_council:
        return _COMMITTEE_SECTION if committee else _FEDERAL_COUNCIL_SECTION
    return None


_SPEAKER_NAMES = {_COMMITTEE_SECTION: "initiative/referendum committee", _FEDERAL_COUNCIL_SECTION: "Federal Council and Parliament"}


def load_parsed_booklet(pdf_path: Union[str, Path], parser: PDFParser) -> Dict[str, Any]:
    """
    Parse a booklet once (pages, passages, full text) and cache the result on disk.
    The cache key is the file's content hash, so it survives a different mount path in the judges' container.
    Cache I/O failures (e.g. a read-only filesystem) silently fall back to parsing.
    """
    pdf_path = Path(pdf_path)
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:16]
    cache_file = config.BOOKLET_CACHE_DIR / f"{digest}_p{config.PASSAGE_CHARS}_v{_PARSE_CACHE_VERSION}.json"
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass

    pages = parser.extract_pages(pdf_path)
    paragraphs = parser.extract_paragraphs(pdf_path, passage_chars=config.PASSAGE_CHARS or None, pages=pages)
    full_text = "\n\n".join(f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages)
    parsed = {"pages": pages, "paragraphs": paragraphs, "full_text": full_text}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_file.parent, delete=False) as temporary:
            json.dump(parsed, temporary, ensure_ascii=False)
        Path(temporary.name).replace(cache_file)
    except OSError:
        pass
    return parsed


def chunk_reference(text: str, max_chars: int = 700, min_chars: int = 80) -> List[str]:
    """Split a reference string into passage-sized chunks (paragraphs, then sentences)."""
    blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces: List[str] = []
    for b in blocks:
        if len(b) <= max_chars:
            pieces.append(b)
            continue
        current = ""
        for sent in re.split(r"(?<=[.!?;])\s+", b):
            if current and len(current) + len(sent) + 1 > max_chars:
                pieces.append(current)
                current = sent
            else:
                current = f"{current} {sent}".strip()
        if current:
            pieces.append(current)
    merged: List[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged or [text.strip()]


class ClaimVerificationEngine:
    def __init__(
        self,
        strategy: str = config.DEFAULT_STRATEGY,
        apertus_client: Optional[ApertusClient] = None,
        prompt_mode: str = config.PROMPT_MODE,
    ):
        self.strategy = strategy
        self.prompt_mode = prompt_mode
        self.client = apertus_client or ApertusClient()
        self.pdf_parser = PDFParser()
        self._source_pages = None
        self._source_pages_lock = threading.Lock()
        self._query_translator = None

    def _get_booklet_data(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        path_str = str(Path(pdf_path).resolve())
        key = path_str
        if config.CACHE_SINGLE_FLIGHT:
            stat = Path(path_str).stat()
            key = (path_str, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, config.PASSAGE_CHARS)
            with _BOOKLET_LOCK_GUARD:
                lock = _BOOKLET_LOCKS.setdefault(key, threading.Lock())
            with lock:
                return self._cached_booklet(key, path_str)
        return self._cached_booklet(key, path_str)

    def _cached_booklet(self, key, path_str):
        if key not in _GLOBAL_BOOKLET_CACHE:
            parsed = load_parsed_booklet(path_str, self.pdf_parser)
            _GLOBAL_BOOKLET_CACHE[key] = {**parsed, "retriever": PassageRetriever(parsed["paragraphs"])}
        return _GLOBAL_BOOKLET_CACHE[key]

    def _finalize_evidence(self, result, vote=None):
        if config.EVIDENCE_POLICY == "legacy" or result.strategy == "direct_reference" or result.error:
            return result
        from src.evidence import SourcePages, construct_evidence
        with self._source_pages_lock:
            if self._source_pages is None:
                self._source_pages = SourcePages(Path("."), lazy=config.SOURCE_PAGES_LAZY)
        # The postprocessor receives baseline-attributed pages and inference inputs
        # (including the baseline's explicit top-page fallback for missing IDs).
        # No expected labels, reference passage or gold evidence enter this path.
        raw = {"id": result.id, "label": result.label,
               "evidence": [{"page": s.page_number, "text": s.quote} for s in result.evidence_sources]}
        case = {"booklet": {"path": str(Path(result.booklet_path).resolve())},
                "claim": {"text": result.claim}, "vote": vote or ""}
        new = construct_evidence(raw, case, self._source_pages, config.EVIDENCE_POLICY)
        proposals = {s.page_number: s.proposal_id for s in result.evidence_sources}
        result.evidence_sources = [EvidenceSource(quote=s["text"], page_number=s["page"],
                                                 proposal_id=proposals.get(s["page"])) for s in new["evidence"]]
        result.evidence = [s.quote for s in result.evidence_sources]
        return result

    def verify_claim(
        self,
        claim: str,
        booklet_pdf: Union[str, Path],
        claim_language: Optional[str] = None,
        strategy: Optional[str] = None,
        top_k: int = config.DEFAULT_TOP_K,
        vote: Optional[str] = None,
        case_id: Optional[str] = None,
        booklet_language: Optional[str] = None,
    ) -> PredictionResult:
        """
        Verify whether the voting booklet entails, contradicts, or is neutral to the claim.
        Supports ADVANCED TASK from Hack Apertus: accepts optional 'vote' title to scope proposal.
        """
        strat = strategy or self.strategy
        booklet_data = self._get_booklet_data(booklet_pdf)
        retrieval_claim, query_variants = claim, None
        translation_usage, query_metadata = (0, 0), {}
        if config.RETRIEVAL_QUERY_MODE != "original":
            if strat != "hybrid" or config.TRANSLATE_CLAIM or self.prompt_mode not in ("ids", "compact"):
                raise ValueError("Query translation requires hybrid ids/compact retrieval and an unchanged NLI claim")
            source_language = booklet_language or guess_language(" ".join(p['text'] for p in booklet_data['pages'][:3]))
            language = claim_language or guess_language(claim)
            query_metadata = {'mode': config.RETRIEVAL_QUERY_MODE, 'claim_language': language,
                              'booklet_language': source_language, 'translation_used': False}
            if language != source_language:
                from src.query_translation import QueryTranslator
                with self._source_pages_lock:
                    if self._query_translator is None:
                        self._query_translator = QueryTranslator(self.client, config.QUERY_TRANSLATION_CACHE_DIR)
                translated, prompt_tokens, completion_tokens, receipt = self._query_translator.get(claim, language, source_language)
                translation_usage = prompt_tokens, completion_tokens
                query_metadata.update(receipt)
                if translated:
                    query_metadata['translation_used'] = True
                    if config.RETRIEVAL_QUERY_MODE == "translated":
                        retrieval_claim = translated
                    else:
                        query_variants = [translated]

        candidate_paras = []
        if strat == "full":
            # Strategy 1: Provide full booklet text to Apertus
            context = booklet_data["full_text"]
            candidate_paras = booklet_data["paragraphs"]
        else:
            # Strategy 2: Retrieve top-k relevant paragraphs (with proposal isolation & vote targeting)
            retriever: PassageRetriever = booklet_data["retriever"]
            if strat == "hybrid":
                # Strategy 3: booklet-wide BM25(claim) + BM25(vote title), no hard proposal filter
                candidate_paras = retriever.retrieve_hybrid(
                    retrieval_claim, top_k=top_k, target_vote=vote,
                    query_variants=query_variants,
                    exclude_sections=opposing_sections(claim) if config.SPEAKER_AWARE else None,
                    ensure_sections={side: config.SPEAKER_BOOST}
                    if config.SPEAKER_BOOST and (side := attributed_section(claim)) else None,
                )
            else:
                candidate_paras = retriever.retrieve(claim, top_k=top_k, target_vote=vote)
            if config.PAGE_MAX_CHARS:
                query = f"{claim} {vote or ''}"
                candidate_paras = [{**p, "text": clip_to_query(p["text"], query, config.PAGE_MAX_CHARS)} for p in candidate_paras]
            context_blocks = []
            for p in candidate_paras:
                context_blocks.append(f"[Page {p['page_number']}] {p['text']}")
            context = "\n\n".join(context_blocks)

        # 1. Deterministic Numerical Conflict Check
        num_conflict: Optional[NumericalConflictResult] = detect_numerical_conflict(claim, context)
        numerical_conflict_msg = num_conflict.explanation if num_conflict else None

        if self.prompt_mode in ("compact", "ids"):
            result = self._compact_predict(
                passages=candidate_paras,
                claim=claim,
                claim_language=claim_language,
                vote=vote,
                case_id=case_id,
                strategy=strat,
                booklet_path=str(booklet_pdf),
                numerical_conflict_msg=numerical_conflict_msg,
            )
            result.tokens_prompt += translation_usage[0]
            result.tokens_completion += translation_usage[1]
            result.tokens_total += sum(translation_usage)
            result.latency_ms += query_metadata.get('remote_ms', 0)
            result.retrieval_query_metadata = query_metadata
            return result

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
        if config.NUMERIC_OVERRIDE and num_conflict and final_label != 2:
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

        return self._finalize_evidence(PredictionResult(
            id=case_id if case_id is not None else "case-0001",
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
            error=nli_output.error,
        ), vote=vote)

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

        if self.prompt_mode in ("compact", "ids"):
            chunks = [{"text": c, "page_number": None, "proposal_id": None} for c in chunk_reference(reference)]
            return self._compact_predict(
                passages=chunks,
                claim=claim,
                claim_language=claim_language,
                vote=None,
                case_id=case_id,
                strategy="direct_reference",
                booklet_path="supplied_reference",
                numerical_conflict_msg=numerical_conflict_msg,
            )

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

        if config.NUMERIC_OVERRIDE and num_conflict and final_label != 2:
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
            ev_sources.append(EvidenceSource(quote=reference, page_number=None, proposal_id=None))
            final_ev = [reference]

        elapsed_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            id=case_id if case_id is not None else "case-0001",
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
            error=nli_output.error,
        )

    def _compact_predict(
        self,
        passages: List[Dict[str, Any]],
        claim: str,
        claim_language: Optional[str],
        vote: Optional[str],
        case_id: Optional[str],
        strategy: str,
        booklet_path: str,
        numerical_conflict_msg: Optional[str],
    ) -> PredictionResult:
        """Compact mode: numbered passages in, '<label>|<passage ids>' out; evidence is verbatim passage text."""
        def label(p: Dict[str, Any]) -> str:
            parts = [f"p. {p['page_number']}"] if p.get("page_number") else []
            if p.get("section"):
                parts.append(p["section"])  # who is speaking: committee vs. Federal Council, legal text, ...
            return f"({'; '.join(parts)}) {p['text']}" if parts else p["text"]

        texts = [label(p) for p in passages]
        model_claim, extra_in, extra_out, extra_ms = claim, 0, 0, 0.0
        if config.TRANSLATE_CLAIM and passages:
            booklet_lang = guess_language(" ".join(p["text"] for p in passages[:3]))
            if (claim_language or guess_language(claim)) != booklet_lang:
                translation, extra_in, extra_out, extra_ms = self.client.translate(claim, booklet_lang)
                if translation:
                    name = self.client.LANGUAGE_NAMES.get(booklet_lang, booklet_lang)
                    model_claim = f"{claim}\n(Translation into {name}: {translation})"
        if config.SPEAKER_HINT and (side := attributed_section(claim)):
            own = [f"P{k}" for k, p in enumerate(passages, 1) if p.get("section") == side]
            if own:
                model_claim = f"{model_claim}\n(Passages written by the {_SPEAKER_NAMES[side]} itself: {', '.join(own)})"
        if self.prompt_mode == "ids":
            out = self.client.infer(context="", claim=model_claim, claim_language=claim_language, passages=texts)
        else:
            out = self.client.infer_compact(texts, model_claim, claim_language=claim_language, vote=vote)
        out.tokens_prompt += extra_in
        out.tokens_completion += extra_out
        out.tokens_total += extra_in + extra_out
        out.latency_ms += extra_ms

        ids = list(out.evidence_ids)
        decision_rule = out.decision_rule_applied
        if out.label != 1 and not ids and passages:
            ids = [1]  # model cited nothing: fall back to the top-ranked passage
            decision_rule = f"{decision_rule or ''} | evidence fallback: top passage".strip(" |")

        sources: List[EvidenceSource] = []
        evidence: List[str] = []
        if out.label != 1:
            for i in ids:
                p = passages[i - 1]
                evidence.append(p["text"])
                sources.append(EvidenceSource(quote=p["text"], page_number=p.get("page_number"), proposal_id=p.get("proposal_id")))

        return self._finalize_evidence(PredictionResult(
            id=case_id if case_id is not None else "case-0001",
            claim=claim,
            label=out.label,
            label_name=config.LABEL_MAPPING.get(out.label, "Unknown"),
            reasoning=out.reasoning,
            evidence=evidence,
            evidence_sources=sources,
            numerical_conflict=numerical_conflict_msg,
            p_entail=out.p_entail,
            p_neutral=out.p_neutral,
            p_contra=out.p_contra,
            fuzzy_rule=decision_rule,
            decision_rule=decision_rule,
            strategy=strategy,
            booklet_path=booklet_path,
            tokens_prompt=out.tokens_prompt,
            tokens_completion=out.tokens_completion,
            tokens_total=out.tokens_total,
            latency_ms=out.latency_ms,
            error=out.error,
            context_pages=list(dict.fromkeys(p["page_number"] for p in passages if p.get("page_number"))),
        ), vote=vote)
