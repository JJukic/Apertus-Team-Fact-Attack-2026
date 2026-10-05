"""Run the unchanged evaluators with a hard API budget and per-case journal.

This is an experiment harness, not an alternative inference pipeline. SDK retries
are disabled; the existing Apertus retry loop shares the same request budget.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.apertus_client import ApertusClient
from src.embeddings import HybridSettings, pdf_hash
from src.evaluator import BenchmarkEvaluator
from src.inference import ClaimVerificationEngine, _GLOBAL_BOOKLET_CACHE


class BudgetExhausted(RuntimeError):
    pass


class BudgetedCreate:
    def __init__(self, create, maximum, journal):
        if maximum < 1:
            raise ValueError("API request budget must be positive")
        self.create = create
        self.maximum = maximum
        self.journal = journal
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, **kwargs):
        if self.requests >= self.maximum:
            raise BudgetExhausted("Authorized API request limit reached")
        self.requests += 1
        start = time.perf_counter()
        event = {"request_number": self.requests, "model_requested": kwargs["model"]}
        try:
            response = self.create(**kwargs)
            usage = response.usage
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            event.update(status="success", response=response.model_dump(mode="json"))
            return response
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            event["latency_ms"] = (time.perf_counter() - start) * 1000
            self.journal(event)


class BenchmarkClient(ApertusClient):
    def __init__(self, maximum, journal):
        super().__init__()
        if self.mock:
            raise ValueError("Live comparison requires the API key and MOCK_APERTUS=false")
        # Each SDK invocation now makes at most one HTTP request.
        self._sdk = self.client.with_options(max_retries=0)
        self.budget = BudgetedCreate(self._sdk.chat.completions.create, maximum, journal)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.budget)))

    def infer(self, *args, **kwargs):
        if self.budget.requests >= self.budget.maximum:
            raise BudgetExhausted("Authorized API request limit reached")
        result = super().infer(*args, **kwargs)
        if result.reasoning.startswith(("API Error", "Response Parse Error")):
            raise RuntimeError("Invalid API result; stop rather than score an operational failure as Neutral")
        return result


class RecordingEngine(ClaimVerificationEngine):
    def __init__(self, records, record_result, **kwargs):
        super().__init__(**kwargs)
        self.records = records
        self.record_result = record_result
        self.record_index = 0

    def verify_claim(self, **kwargs):
        record = self.records[self.record_index % len(self.records)]
        # Budget check before PDF/retrieval work as well as before the actual request.
        if self.client.budget.requests >= self.client.budget.maximum:
            raise BudgetExhausted("Authorized API request limit reached")
        result = super().verify_claim(**kwargs)
        self.record_result({"case_index": self.record_index % len(self.records),
                            "dataset_record": record, "prediction": result.model_dump(mode="json")})
        self.record_index += 1
        return result


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-api-requests", required=True, type=int)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not records or any(r.get("split") not in ("dev", "validation") for r in records):
        parser.error("Explicitly marked dev/validation cases are required")
    if args.max_api_requests < 4 * len(records):
        parser.error("Budget is insufficient for two methods with two passes each")
    if args.output.exists():
        parser.error("Output already exists; do not accidentally repeat a paid run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    api_journal = args.output.with_suffix(".api.jsonl")
    case_journal = args.output.with_suffix(".cases.jsonl")
    if api_journal.exists() or case_journal.exists():
        parser.error("Experiment journals already exist; paid runs must not be repeated silently")
    phase = {}
    payload = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()), "dataset_sha256": pdf_hash(args.dataset),
        "sample_count": len(records), "mock_mode": False,
        "authorized_max_api_requests": args.max_api_requests,
        "model": config.LLM_NAME, "platform": platform.platform(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(HybridSettings()).items()},
        "source_sha256": {name: pdf_hash(Path(__file__).resolve().parents[1] / "src" / name)
                          for name in ["inference.py", "retriever.py", "hybrid_retriever.py", "pdf_parser.py",
                                       "apertus_client.py", "evaluator.py", "embeddings.py", "config.py"]},
        "booklet_sha256": {lang: pdf_hash(config.BOOKLETS_DIR / f"2024-11-24_{lang}.pdf")
                           for lang in sorted({r["booklet_language"] for r in records})},
        "phase_note": "First pass includes existing disk hits or new index construction; warm_repeat reuses the same engine. No caches are deleted.",
        "reports": {},
    }

    def journal(path, event):
        with path.open("a") as handle:
            handle.write(json.dumps({**phase, **event}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    client = BenchmarkClient(args.max_api_requests, lambda event: journal(api_journal, event))
    start = time.perf_counter()

    def checkpoint():
        payload.update(api_requests_used=client.budget.requests,
                       reported_api_input_tokens=client.budget.input_tokens,
                       reported_api_output_tokens=client.budget.output_tokens,
                       wall_time_seconds=time.perf_counter() - start,
                       current_phase=dict(phase))
        save_json(args.output, payload)

    checkpoint()
    try:
        for strategy in ("hybrid", "hybrid_dense"):
            _GLOBAL_BOOKLET_CACHE.clear()
            engine = RecordingEngine(records, lambda event: journal(case_journal, event),
                                     strategy=strategy, apertus_client=client)
            evaluator = BenchmarkEvaluator(engine)
            payload["reports"][strategy] = {}
            for pass_name in ("first_pass", "warm_repeat"):
                phase.update(strategy=strategy, phase=pass_name)
                checkpoint()
                report = evaluator.evaluate(args.dataset, strategy)
                payload["reports"][strategy][pass_name] = report
                checkpoint()
        payload["status"] = "complete"
    except BaseException as exc:
        payload.update(status="aborted", error_type=type(exc).__name__)
        checkpoint()
        raise
    finally:
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    print(f"Saved {args.output}; API requests: {client.budget.requests}/{client.budget.maximum}")


if __name__ == "__main__":
    main()
