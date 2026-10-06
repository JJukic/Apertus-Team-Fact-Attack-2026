"""Prepare a frozen-context prompt experiment offline, then run it with a new budget.

Preparation never constructs a live client. Execution sends only claim, language
and prepared context to the existing pipeline; annotations stay in scoring.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src import config
from src.apertus_client import ApertusClient
from src.embeddings import HybridSettings, pdf_hash
from src.evidence_matcher import match_quote
from src.inference import ClaimVerificationEngine
from src.pdf_parser import PDFParser
from sklearn.metrics import confusion_matrix, f1_score
from run_hybrid_quality_comparison import BenchmarkClient, save_json


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def implementation_hashes():
    return {str(p.relative_to(BASE)): pdf_hash(p)
            for p in [*sorted((BASE / "src").glob("*.py")), Path(__file__),
                      BASE / "scripts/run_hybrid_quality_comparison.py"]}


class PromptClient(BenchmarkClient):
    """Inject the experimental suffix at the transport boundary, only if selected."""
    def __init__(self, maximum, journal, addendum):
        super().__init__(maximum, journal)
        self.addendum = addendum
        self.variant = "baseline"
        self.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create_with_prompt)))

    def create_with_prompt(self, **kwargs):
        messages = deepcopy(kwargs["messages"])
        if self.variant not in ("baseline", "candidate"):
            raise ValueError("Unknown prompt variant")
        if self.variant == "candidate":
            if not messages or messages[0]["role"] != "system":
                raise ValueError("Expected an existing system prompt")
            messages[0]["content"] += "\n\n" + self.addendum
        return self.budget(**{**kwargs, "messages": messages})


class FrozenEngine(ClaimVerificationEngine):
    def retrieve_context(self, *args, **kwargs):
        # The same immutable snapshot is used by both prompt variants.
        return deepcopy(self.frozen)


def summarize(rows):
    def metrics(items, key):
        gold = [r["gold"] for r in items]
        labels = [r[key]["label"] for r in items]
        return {"count": len(items),
                "macro_f1": float(f1_score(gold, labels, labels=[0, 1, 2],
                                           average="macro", zero_division=0)),
                "confusion_matrix_labels_0_1_2": confusion_matrix(gold, labels, labels=[0, 1, 2]).tolist(),
                "source_quotes": sum(r[key]["source_quotes"] for r in items),
                "exact_source_matches": sum(r[key]["exact_source_matches"] for r in items),
                "semantic_evidence_validity": "pending_manual_review"}
    return {"baseline": metrics(rows, "baseline"), "candidate": metrics(rows, "candidate"),
            "by_language": {lang: {v: metrics([r for r in rows if r["language"] == lang], v)
                                  for v in ("baseline", "candidate")}
                            for lang in sorted({r["language"] for r in rows})},
            "corrected": sum(r["baseline"]["label"] != r["gold"] == r["candidate"]["label"] for r in rows),
            "regressed": sum(r["baseline"]["label"] == r["gold"] != r["candidate"]["label"] for r in rows),
            "changed_cases": [r["id"] for r in rows if r["baseline"]["label"] != r["candidate"]["label"]]}


def prepare(dataset, prompt, methods, output):
    if output.exists():
        raise ValueError("Plan already exists; use a new path for a different experiment")
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Nonempty dataset with unique case IDs required")
    if any(r.get("split") not in ("dev", "validation", "test") for r in records):
        raise ValueError("Each case must explicitly declare its split")
    if len({r["split"] for r in records}) != 1:
        raise ValueError("Keep development and held-out experiments separate")
    if any(type(r["entailment_label"]) is not int or r["entailment_label"] not in (0, 1, 2) for r in records):
        raise ValueError("Gold labels must be integers 0, 1, 2")
    addendum = prompt.read_text().strip()
    if not addendum:
        raise ValueError("Empty candidate prompt")
    # Dense model downloads and silent fallback are disabled for this experiment.
    settings = HybridSettings(local_files_only=True, fallback=False)
    engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True), hybrid_settings=settings)
    snapshots, pdfs, paragraphs = [], {}, {}
    for method in methods:
        for record in records:
            pdf = config.BOOKLETS_DIR / f'{record["booklet_date"]}_{record["booklet_language"]}.pdf'
            if not pdf.exists():
                raise FileNotFoundError(pdf)
            pdfs[str(pdf.resolve())] = pdf_hash(pdf)
            vote = record.get("vote")
            if isinstance(vote, dict):
                vote = vote.get("title") or vote.get("text")
            frozen = engine.retrieve_context(record["claim"], pdf, strategy=method, vote=vote)
            if frozen["metrics"]["actual_strategy"] != method:
                raise ValueError("Retrieval fallback invalidates the comparison")
            paragraphs[str(pdf.resolve())] = frozen["booklet_data"]["paragraphs"]
            # Runtime booklet data also contains a non-serializable retriever.
            # Store only the fields used downstream, deduplicating paragraphs.
            frozen = {k: frozen[k] for k in ("context", "sources", "metrics")}
            snapshots.append({"record": record, "method": method, "pdf": str(pdf.resolve()),
                              "context_sha256": text_hash(frozen["context"]), "prepared": frozen})
    plan = {"status": "prepared_no_api_calls", "dataset": str(dataset.resolve()),
            "dataset_sha256": pdf_hash(dataset), "prompt": str(prompt.resolve()),
            "prompt_sha256": pdf_hash(prompt), "addendum": addendum,
            "model": config.LLM_NAME, "base_url": config.LLM_BASE_URL,
            "implementation_sha256": implementation_hashes(), "pdf_sha256": pdfs,
            "methods": methods, "cases_per_method": len(records),
            "minimum_api_requests": 2 * len(snapshots),
            "split": records[0]["split"], "labels": dict(Counter(r["entailment_label"] for r in records)),
            "settings": {k: str(v) for k, v in vars(settings).items()},
            "snapshots": snapshots, "booklet_paragraphs": paragraphs,
            "cost_note": "No endpoint prices are documented in this plan; token usage is recorded live.",
            "quality_gate": "F1 gain alone is insufficient: manually review evidence and confirm on unused test data."}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, plan)
    print(f"Prepared {len(snapshots)} frozen contexts; minimum {plan['minimum_api_requests']} new API attempts. No API calls.")


def validate_plan(plan):
    if plan["implementation_sha256"] != implementation_hashes():
        raise ValueError("Implementation changed after preparation")
    for name, expected in {plan["dataset"]: plan["dataset_sha256"],
                           plan["prompt"]: plan["prompt_sha256"], **plan["pdf_sha256"]}.items():
        if pdf_hash(Path(name)) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    if plan["model"] != config.LLM_NAME or plan["base_url"] != config.LLM_BASE_URL:
        raise ValueError("Model or endpoint changed after preparation")
    if plan["addendum"] != Path(plan["prompt"]).read_text().strip():
        raise ValueError("Candidate text differs from frozen prompt")
    for item in plan["snapshots"]:
        if text_hash(item["prepared"]["context"]) != item["context_sha256"]:
            raise ValueError("Frozen context changed")


def execute(plan_path, output, maximum, approved_hash):
    if pdf_hash(plan_path) != approved_hash:
        raise ValueError("Execution requires the exact approved plan hash")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if maximum < plan["minimum_api_requests"]:
        raise ValueError("Budget insufficient for the planned paired comparison")
    api_path, case_path = output.with_suffix(".api.jsonl"), output.with_suffix(".cases.jsonl")
    if any(p.exists() for p in (output, api_path, case_path)):
        raise ValueError("Existing artifacts prevent accidental paid repetition")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    def journal(path, event):
        with path.open("a") as handle:
            handle.write(json.dumps({**state, **event}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    client = PromptClient(maximum, lambda e: journal(api_path, e), plan["addendum"])
    engine = FrozenEngine(apertus_client=client)
    report = {"status": "running", "plan_sha256": approved_hash, "authorized_max_api_requests": maximum,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "results": [], "reports": {}}
    pages = {}
    def checkpoint():
        report.update(api_requests_used=client.budget.requests,
                      reported_input_tokens=client.budget.input_tokens,
                      reported_output_tokens=client.budget.output_tokens)
        save_json(output, report)
    checkpoint()
    try:
        for index, item in enumerate(plan["snapshots"]):
            record = item["record"]
            pdf = item["pdf"]
            engine.frozen = {**item["prepared"],
                             "booklet_data": {"paragraphs": plan["booklet_paragraphs"][pdf]}}
            if pdf not in pages:
                pages[pdf] = PDFParser().extract_pages(pdf)
            row = {"id": record["id"], "method": item["method"],
                   "language": record["claim_language"], "gold": record["entailment_label"]}
            # Alternate order to reduce systematic endpoint/time effects.
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for variant in order:
                state.update(case_id=record["id"], method=item["method"], variant=variant,
                             context_sha256=item["context_sha256"])
                client.variant = variant
                pred = engine.verify_claim(record["claim"], pdf, record["claim_language"], item["method"])
                matched = sum(bool(match_quote(s.quote, pages[pdf])) for s in pred.evidence_sources)
                row[variant] = {"label": pred.label, "source_quotes": len(pred.evidence_sources),
                                "exact_source_matches": matched, "prediction": pred.model_dump(mode="json")}
                journal(case_path, {"record": record, **row[variant]})
                checkpoint()
            report["results"].append(row)
            checkpoint()
        report["reports"] = {m: summarize([r for r in report["results"] if r["method"] == m]) for m in plan["methods"]}
        report["status"] = "complete_pending_evidence_review"
    except BaseException as exc:
        report.update(status="aborted", error_type=type(exc).__name__)
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--prompt", type=Path, required=True)
    prep.add_argument("--methods", nargs="+", choices=config.STRATEGIES, default=["retrieval", "hybrid_dense"])
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--approved-plan-sha256", required=True)
    run.add_argument("--max-api-requests", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if len(set(args.methods)) != len(args.methods):
            parser.error("Duplicate methods")
        prepare(args.dataset, args.prompt, args.methods, args.output)
    else:
        execute(args.plan, args.output, args.max_api_requests, args.approved_plan_sha256)


if __name__ == "__main__":
    main()
