"""
Benchmark Evaluator for Track 2A (OST).

Mirrors the official judging criteria:
- Macro-F1 over Entailment (0), Neutral (1), Contradiction (2)  [primary]
- Breakdown by claim -> reference language pair (incl. cross-lingual pairs)
- Evidence grounding: do returned passages come from the gold reference section?
- Efficiency: input/output tokens, mean and p95 inference time

Supports both tasks:
- advanced: claim + booklet PDF (+ vote title)
- beginner: claim + reference_string

Every run is saved to results/ with model, strategy, git commit and per-sample rows.
"""

import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tabulate import tabulate
from tqdm import tqdm

from src import config
from src.inference import ClaimVerificationEngine, PredictionResult

RESULTS_DIR = config.BASE_DIR / "results"
LABELS = [0, 1, 2]
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL benchmark and normalise legacy and official HF schemas."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            claim_lang = r.get("claim_language", "de")
            ref_lang = r.get("reference_language") or r.get("booklet_language") or claim_lang
            booklet_file = r.get("booklet_file")
            if not booklet_file:
                # Booklets are stored by publish date, but some rows carry a publish date that differs from the
                # vote date the file is named after (demo set: published 2026-05-28, file 2026-06-14_*.pdf, as in the URL)
                dates = [r.get("booklet_date") or str(r.get("booklet_publish_date", "2026-06-14"))[:10]]
                dates += re.findall(r"\d{4}-\d{2}-\d{2}", r.get("booklet_url") or "")
                names = [f"{d}_{ref_lang}.pdf" for d in dates]
                booklet_file = next((n for n in names if (config.BOOKLETS_DIR / n).exists()), names[0])
            records.append({
                "id": r.get("id", f"case-{i:04d}"),
                "claim": r["claim"],
                "claim_language": claim_lang,
                "reference_language": ref_lang,
                "reference_string": (r.get("reference_string") or "").strip(),
                "label": int(r["entailment_label"]),
                "vote": r.get("vote"),
                "booklet_pdf": config.BOOKLETS_DIR / booklet_file,
            })
    return records


def stratified_sample(records: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Label-balanced, deterministic sample of n records."""
    if n >= len(records):
        return records
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    for group in by_label.values():
        rng.shuffle(group)
    sample, keys = [], sorted(by_label)
    while len(sample) < n:
        for k in keys:
            if by_label[k] and len(sample) < n:
                sample.append(by_label[k].pop())
    return sample


def evidence_grounded(evidence: List[str], reference: str, threshold: float = 0.6) -> bool:
    """True if any evidence passage is (mostly) contained in the gold reference section."""
    ref_words = set(_WORD_RE.findall(reference.lower()))
    if not ref_words:
        return False
    for ev in evidence:
        ev_words = _WORD_RE.findall(ev.lower())
        if len(ev_words) >= 3 and sum(w in ref_words for w in ev_words) / len(ev_words) >= threshold:
            return True
    return False


def _git_commit() -> str:
    # Inside the Docker image there is no .git: the commit is baked in at build time (GIT_COMMIT build arg)
    fallback = os.getenv("GIT_COMMIT") or "unknown"
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.BASE_DIR)
        if out.returncode != 0 or not out.stdout.strip():
            return fallback
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=config.BASE_DIR)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return fallback


def _macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    return round(float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)), 4)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


class BenchmarkEvaluator:
    def __init__(self, engine: Optional[ClaimVerificationEngine] = None):
        self.engine = engine or ClaimVerificationEngine()

    def _predict(self, rec: Dict[str, Any], task: str, strategy: str, top_k: int) -> PredictionResult:
        if task == "beginner":
            return self.engine.verify_premise(
                claim=rec["claim"],
                reference=rec["reference_string"],
                claim_language=rec["claim_language"],
                case_id=rec["id"],
            )
        return self.engine.verify_claim(
            claim=rec["claim"],
            booklet_pdf=rec["booklet_pdf"],
            claim_language=rec["claim_language"],
            strategy=strategy,
            top_k=top_k,
            vote=rec["vote"],
            case_id=rec["id"],
        )

    def evaluate(
        self,
        dataset_path: Optional[Path] = None,
        strategy: str = config.DEFAULT_STRATEGY,
        limit: Optional[int] = None,
        task: str = "advanced",
        top_k: int = config.DEFAULT_TOP_K,
        workers: int = 1,
        seed: int = 42,
        save: bool = True,
        tag: str = "",
    ) -> Dict[str, Any]:
        path = Path(dataset_path) if dataset_path else config.BENCHMARK_PATH
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at: {path}")

        records = load_records(path)
        if limit:
            records = stratified_sample(records, limit, seed)

        if task == "advanced":
            missing = sorted({str(r["booklet_pdf"].name) for r in records if not r["booklet_pdf"].exists()})
            if missing:
                raise FileNotFoundError(f"Missing booklets (run `python -m src.hf_dataset`): {missing}")
            # Parse every booklet once up front so PDF parsing is not counted as model latency
            for pdf in tqdm(sorted({r["booklet_pdf"] for r in records}), desc="Parsing booklets"):
                self.engine._get_booklet_data(pdf)

        print(f"\nEvaluating {len(records)} samples | task={task} | strategy={strategy} | model={self.engine.client.model_name}")

        started = time.time()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            preds = list(tqdm(
                pool.map(lambda r: self._predict(r, task, strategy, top_k), records),
                total=len(records),
                desc="Evaluating",
            ))
        wall_s = time.time() - started

        report = self._build_report(records, preds, task, strategy, top_k, workers, seed, path, wall_s, tag)
        self.print_report(report)
        if save:
            report["saved_to"] = str(self.save_report(report))
            print(f"Saved -> {report['saved_to']}\n")
        return report

    def _build_report(self, records, preds, task, strategy, top_k, workers, seed, path, wall_s, tag) -> Dict[str, Any]:
        y_true = [r["label"] for r in records]
        y_pred = [p.label for p in preds]
        pairs = [f"{r['claim_language']}->{r['reference_language']}" for r in records]

        by_pair: Dict[str, Any] = {}
        for pair in sorted(set(pairs)):
            idx = [i for i, p in enumerate(pairs) if p == pair]
            yt, yp = [y_true[i] for i in idx], [y_pred[i] for i in idx]
            by_pair[pair] = {
                "count": len(idx),
                "macro_f1": _macro_f1(yt, yp),
                "accuracy": round(sum(a == b for a, b in zip(yt, yp)) / len(idx), 4),
            }

        mono = [i for i, p in enumerate(pairs) if p.split("->")[0] == p.split("->")[1]]
        cross = [i for i in range(len(pairs)) if i not in set(mono)]

        def subset_f1(idx: List[int]) -> Optional[float]:
            return _macro_f1([y_true[i] for i in idx], [y_pred[i] for i in idx]) if idx else None

        # Evidence: only labels 0/2 require evidence
        ev_required = [i for i, p in enumerate(preds) if p.label != 1]
        ev_given = [i for i in ev_required if preds[i].evidence]
        ev_grounded = [i for i in ev_given if evidence_grounded(preds[i].evidence, records[i]["reference_string"])]

        latencies = [p.latency_ms for p in preds]
        errors = [p for p in preds if p.error]

        rows = []
        for r, p in zip(records, preds):
            rows.append({
                "id": r["id"],
                "pair": f"{r['claim_language']}->{r['reference_language']}",
                "vote": r["vote"],
                "claim": r["claim"],
                "true_label": r["label"],
                "pred_label": p.label,
                "correct": r["label"] == p.label,
                "decision_rule": p.decision_rule,
                "probs": [p.p_entail, p.p_neutral, p.p_contra],
                "numerical_conflict": p.numerical_conflict,
                "reasoning": p.reasoning,
                "evidence": p.evidence,
                "evidence_pages": [s.page_number for s in p.evidence_sources],
                "evidence_grounded": evidence_grounded(p.evidence, r["reference_string"]) if p.evidence else None,
                "tokens_prompt": p.tokens_prompt,
                "tokens_completion": p.tokens_completion,
                "latency_ms": p.latency_ms,
                "error": p.error,
                "extracted_statements": p.extracted_statements,
                "stage_warnings": p.stage_warnings,
            })

        return {
            "meta": {
                # With the UTC offset: Docker containers run in UTC, local runs in local time
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "git_commit": _git_commit(),
                "model": self.engine.client.model_name,
                "mock": self.engine.client.mock,
                "task": task,
                "strategy": strategy if task == "advanced" else "direct_reference",
                "top_k": top_k,
                "prompt_mode": self.engine.prompt_mode,
                "passage_chars": config.PASSAGE_CHARS if task == "advanced" else None,
                "ids_reason": config.IDS_REASON,
                "thinking": config.THINKING,
                "thinking_budget": config.THINKING_BUDGET,
                "translate_claim": config.TRANSLATE_CLAIM,
                "page_max_chars": config.PAGE_MAX_CHARS,
                "speaker_boost": config.SPEAKER_BOOST,
                "speaker_hint": config.SPEAKER_HINT,
                "speaker_aware": config.SPEAKER_AWARE,
                "few_shot": config.FEW_SHOT,
                "dataset": str(path.name),
                "seed": seed,
                "workers": workers,
                "tag": tag,
                "wall_time_s": round(wall_s, 1),
            },
            "sample_count": len(records),
            "macro_f1": _macro_f1(y_true, y_pred),
            "accuracy": round(sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(records)), 4),
            "macro_f1_monolingual": subset_f1(mono),
            "macro_f1_crosslingual": subset_f1(cross),
            "by_language_pair": by_pair,
            "evidence": {
                "predictions_needing_evidence": len(ev_required),
                "evidence_given_rate": round(len(ev_given) / len(ev_required), 4) if ev_required else None,
                "evidence_grounded_rate": round(len(ev_grounded) / len(ev_given), 4) if ev_given else None,
            },
            "efficiency": {
                "avg_prompt_tokens": round(sum(p.tokens_prompt for p in preds) / max(1, len(preds)), 1),
                "avg_completion_tokens": round(sum(p.tokens_completion for p in preds) / max(1, len(preds)), 1),
                "latency_mean_ms": round(sum(latencies) / max(1, len(latencies)), 1),
                "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
            },
            "errors": {"count": len(errors), "examples": [e.error for e in errors[:5]]},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
            "detailed_classification": classification_report(
                y_true, y_pred, labels=LABELS,
                target_names=["0: Entailment", "1: Neutral", "2: Contradiction"],
                output_dict=True, zero_division=0,
            ),
            "results": rows,
        }

    @staticmethod
    def save_report(report: Dict[str, Any]) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        m = report["meta"]
        model_short = m["model"].split("/")[-1]
        stamp = m["timestamp"][:19].replace(":", "").replace("-", "")
        name = f"{stamp}_{m['task']}_{m['strategy']}_{m.get('prompt_mode', 'json')}_{model_short}{('_' + m['tag']) if m['tag'] else ''}.json"
        out = RESULTS_DIR / name
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @staticmethod
    def print_report(report: Dict[str, Any]):
        m = report["meta"]
        print("\n" + "=" * 64)
        print("              TRACK 2A (OST) BENCHMARK REPORT")
        print("=" * 64)
        print(f"Task / Strategy:     {m['task']} / {m['strategy']} / prompt={m.get('prompt_mode')}   (model: {m['model']}{', MOCK' if m['mock'] else ''})")
        print(f"Samples:             {report['sample_count']}   dataset: {m['dataset']}   commit: {m['git_commit']}")
        print(f"PRIMARY  Macro-F1:   {report['macro_f1']:.4f}   (accuracy {report['accuracy']:.4f})")
        print(f"  monolingual F1:    {report['macro_f1_monolingual']}")
        print(f"  cross-lingual F1:  {report['macro_f1_crosslingual']}")
        ev = report["evidence"]
        print(f"Evidence given:      {ev['evidence_given_rate']}   grounded in gold section: {ev['evidence_grounded_rate']}")
        eff = report["efficiency"]
        print(f"Tokens in / out:     {eff['avg_prompt_tokens']} / {eff['avg_completion_tokens']}")
        print(f"Latency mean / p95:  {eff['latency_mean_ms']} ms / {eff['latency_p95_ms']} ms")
        if report["errors"]["count"]:
            print(f"ERRORS:              {report['errors']['count']}  e.g. {report['errors']['examples'][:2]}")
        print("-" * 64)

        det = report["detailed_classification"]
        rows = []
        for lbl in ["0: Entailment", "1: Neutral", "2: Contradiction"]:
            d = det.get(lbl, {})
            rows.append([lbl, f"{d.get('precision', 0):.2f}", f"{d.get('recall', 0):.2f}", f"{d.get('f1-score', 0):.2f}", d.get("support", 0)])
        print(tabulate(rows, headers=["Class", "Precision", "Recall", "F1", "Support"], tablefmt="github"))

        print("\nConfusion matrix (rows = true, cols = predicted 0/1/2):")
        for lbl, row in zip(LABELS, report["confusion_matrix"]):
            print(f"  {lbl}: {row}")

        print("\nBy language pair (claim -> reference):")
        pair_rows = [[k, v["count"], f"{v['macro_f1']:.3f}", f"{v['accuracy']:.3f}"] for k, v in report["by_language_pair"].items()]
        print(tabulate(pair_rows, headers=["Pair", "N", "Macro-F1", "Acc"], tablefmt="github"))
        print("=" * 64)
