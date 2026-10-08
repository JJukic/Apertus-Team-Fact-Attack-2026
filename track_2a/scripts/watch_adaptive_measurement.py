"""Write a lightweight live F1 table and finalize a running validation job.

Never starts or repeats inference. Complete predictions trigger the existing
comparison summarizer; incomplete results remain explicitly provisional.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.adaptive_metrics import language_metrics


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def progress(folder):
    manifest = json.loads((folder / "run.json").read_text())
    source = BASE / "data/evaluation/adaptive/validation.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["signature"]["split_sha256"]:
        raise ValueError("Validation data changed")
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    reports = {}
    for name in manifest["signature"]["profiles"]:
        rows = []
        for case in cases:
            path = folder / "predictions" / name / f"{case['id']}.json"
            row = json.loads(path.read_text()) if path.exists() else {"label": None}
            if path.exists() and row["entailment_label"] != case["entailment_label"]:
                raise ValueError("Prediction gold metadata differs from validation data")
            rows.append({"entailment_label": case["entailment_label"],
                         "claim_language": case["claim_language"],
                         "booklet_language": case["booklet_language"], "label": row.get("label")})
        reports[name] = language_metrics(rows)
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "expected_cases": len(cases), "prepared_cases": len(list((folder / "prepared").glob("*.json"))),
              "split": "validation", "model": manifest["signature"]["model"],
              "note": "Incomplete scores use successful predictions only and are provisional.",
              "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "profiles": reports}
    write(folder / "progress.json", json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = ["# Live validation measurement", "", "| Profile | Successful cases | Macro-F1 | State |",
             "|---|---:|---:|---|"]
    for name, report in reports.items():
        m = report["overall"]
        score = f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "pending"
        state = "complete validation" if m["eligible_for_selection"] else "provisional" if m["successful_predictions"] else "pending"
        lines.append(f"| {name} | {m['successful_predictions']}/{len(cases)} | {score} | {state} |")
    lines += ["", result["note"], "Validation scores are not held-out test scores."]
    write(folder / "progress.md", "\n".join(lines)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    folder = args.output / "validation"
    reported_adaptive = False
    while True:
        report = progress(folder)
        completed = report["profiles"].get("E6", {}).get("overall", {}).get("eligible_for_selection", False)
        try:
            os.kill(args.pid, 0)
            running = True
        except ProcessLookupError:
            running = False
        if (completed and not reported_adaptive) or not running:
            manifest = json.loads((folder / "run.json").read_text())
            env = os.environ.copy()
            env["EMBEDDING_BATCH_SIZE"] = str(manifest["signature"]["dense"]["batch_size"])
            env["EMBEDDING_CACHE_DIR"] = manifest["signature"]["dense"]["cache_dir"]
            env["EMBEDDING_MAX_LENGTH"] = str(manifest["signature"]["dense"]["max_length"])
            subprocess.run([sys.executable, str(BASE / "scripts/run_adaptive_comparison.py"),
                            "summarize", "--output", str(args.output), "--device",
                            manifest["signature"]["settings"]["device"]], check=True, env=env)
            reported_adaptive = True
        if not running:
            print("Observed job exited; saved actual coverage and scores. No inference restarted.", flush=True)
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
