"""Recalculate historical metrics without running inference or changing archives.

This is an archive audit, not a fresh baseline reproduction. Historical evidence
grounding and API latency are not substitutes for official Hit@5 or non-LLM time.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics

from sklearn.metrics import classification_report, f1_score


FILES = {
    "A": "20261006T142147_advanced_hybrid_ids_Apertus-v1.5-70B-thinking_final.json",
    "B": "20261006T132045_beginner_direct_reference_ids_Apertus-v1.5-70B-thinking_final_beginner.json",
}


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    archive = json.loads(raw)
    rows = archive["results"]
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or len(rows) != archive["sample_count"]:
        raise ValueError(f"Invalid case population: {path}")
    gold = [r["true_label"] for r in rows]
    predictions = [r["pred_label"] for r in rows]
    if any(type(label) is not int or label not in (0, 1, 2) for label in gold + predictions):
        raise ValueError(f"Invalid label in archive: {path}")
    score = float(f1_score(gold, predictions, labels=[0, 1, 2], average="macro", zero_division=0))
    if abs(score - archive["macro_f1"]) > 0.000051:
        raise ValueError(f"Stored F1 disagrees with case predictions: {path}")

    def subgroup(indices):
        return float(f1_score([gold[i] for i in indices], [predictions[i] for i in indices],
                             labels=[0, 1, 2], average="macro", zero_division=0))

    by_pair = {}
    for pair in sorted({r["pair"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["pair"] == pair]
        by_pair[pair] = {"cases": len(indices), "macro_f1": subgroup(indices)}
    cross = [i for i, r in enumerate(rows) if len(set(r["pair"].split("->"))) == 2]

    def token_summary(key):
        values = [r[key] for r in rows]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"Invalid token count: {path}")
        ordered = sorted(values)
        return {"sum": sum(values), "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": ordered[round(0.95 * (len(ordered) - 1))]}

    return {
        "measurement_type": "historical_prediction_recalculation",
        "fresh_api_run": False,
        "source_file": path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "historical_meta": archive["meta"],
        "cases": len(rows),
        "macro_f1": score,
        "accuracy": sum(a == b for a, b in zip(gold, predictions)) / len(rows),
        "per_class": classification_report(gold, predictions, labels=[0, 1, 2],
                                           output_dict=True, zero_division=0),
        "language_pair_direction": "claim->reference",
        "by_language_pair": by_pair,
        "cross_language_macro_f1": subgroup(cross) if cross else None,
        "reported_error_count": sum(bool(r.get("error")) for r in rows),
        "input_tokens_reported": token_summary("tokens_prompt"),
        "output_tokens_reported": token_summary("tokens_completion"),
        "official_evidence_hit5": None,
        "non_llm_processing_time": None,
        "limitations": [
            "Historical test dates were subsequently used in architecture selection.",
            "Historical word-overlap evidence metric is not official Evidence Hit@5.",
            "Reported token counts cannot establish accounting for every failed API attempt.",
            "Archive contains no request intervals for measuring the union of in-flight requests.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "historical_archive_verified_fresh_reproduction_pending",
        "starting_commit": "a0e9e772dd388bdff4b8635ef0e09872ef705af6",
        "audit_python": platform.python_version(),
        "tasks": {task: summarize(args.results_dir / name) for task, name in FILES.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for task, data in report["tasks"].items():
        print(f"{task}: {data['cases']} cases, Macro-F1 {data['macro_f1']:.6f} (historical archive)")


if __name__ == "__main__":
    main()
