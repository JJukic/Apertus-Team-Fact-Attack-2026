"""
Summarise saved benchmark runs (results/*.json) into results/summary.json.

The Streamlit app and the README read their numbers from here instead of hard-coding them.
The headline run per task is the latest test-split run whose settings match the current defaults.

Usage:
    python -m src.results_summary
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config

RESULTS_DIR = config.BASE_DIR / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"


def _row(report: Dict[str, Any], path: Path) -> Dict[str, Any]:
    m = report["meta"]
    eff = report["efficiency"]
    return {
        "file": path.name,
        "timestamp": m["timestamp"],
        "split": Path(m["dataset"]).stem,
        "task": m["task"],
        "strategy": m["strategy"],
        "prompt_mode": m.get("prompt_mode", "json"),
        "top_k": m.get("top_k"),
        "passage_chars": m.get("passage_chars"),
        "ids_reason": bool(m.get("ids_reason", "reason" in m.get("tag", ""))),
        "thinking": bool(m.get("thinking", False)),
        "translate_claim": bool(m.get("translate_claim", False)),
        "page_max_chars": m.get("page_max_chars", 0) or 0,
        "speaker_boost": m.get("speaker_boost", 0) or 0,
        "speaker_hint": bool(m.get("speaker_hint", False)),
        "model": m["model"],
        "tag": m.get("tag", ""),
        "n": report["sample_count"],
        "macro_f1": report["macro_f1"],
        "macro_f1_crosslingual": report.get("macro_f1_crosslingual"),
        "macro_f1_monolingual": report.get("macro_f1_monolingual"),
        "evidence_grounded_rate": report["evidence"].get("evidence_grounded_rate"),
        "avg_prompt_tokens": eff["avg_prompt_tokens"],
        "avg_completion_tokens": eff["avg_completion_tokens"],
        "latency_mean_ms": eff["latency_mean_ms"],
        "latency_p95_ms": eff["latency_p95_ms"],
        "errors": report["errors"]["count"],
        "by_language_pair": report["by_language_pair"],
    }


def load_runs(results_dir: Path = RESULTS_DIR) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == SUMMARY_PATH.name:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("meta", {}).get("mock"):
                runs.append(_row(report, path))
        except (KeyError, json.JSONDecodeError):
            continue
    return runs


def _matches_defaults(run: Dict[str, Any]) -> bool:
    if run["prompt_mode"] != config.PROMPT_MODE or run["model"] != config.LLM_NAME:
        return False
    if run["prompt_mode"] == "ids" and (run["ids_reason"] != config.IDS_REASON or run["thinking"] != config.THINKING):
        return False
    if run["translate_claim"] != config.TRANSLATE_CLAIM:
        return False
    if run["task"] == "beginner":
        return True
    return (
        run["strategy"] == config.DEFAULT_STRATEGY
        and run["top_k"] == config.DEFAULT_TOP_K
        and (run["passage_chars"] or 0) == config.PASSAGE_CHARS
        and run["page_max_chars"] == config.PAGE_MAX_CHARS
        and run["speaker_boost"] == config.SPEAKER_BOOST
        and run["speaker_hint"] == config.SPEAKER_HINT
    )


def headline(runs: List[Dict[str, Any]], task: str, split: str = "test") -> Optional[Dict[str, Any]]:
    candidates = [r for r in runs if r["task"] == task and r["split"] == split and _matches_defaults(r)]
    return max(candidates, key=lambda r: (r["n"], r["timestamp"])) if candidates else None


def build_summary(results_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    runs = load_runs(results_dir)
    return {
        "defaults": {
            "model": config.LLM_NAME,
            "strategy": config.DEFAULT_STRATEGY,
            "top_k": config.DEFAULT_TOP_K,
            "passage_chars": config.PASSAGE_CHARS,
            "prompt_mode": config.PROMPT_MODE,
            "ids_reason": config.IDS_REASON,
            "thinking": config.THINKING,
            "translate_claim": config.TRANSLATE_CLAIM,
            "page_max_chars": config.PAGE_MAX_CHARS,
            "speaker_boost": config.SPEAKER_BOOST,
            "speaker_hint": config.SPEAKER_HINT,
        },
        "headline": {task: headline(runs, task) for task in ("advanced", "beginner")},
        "runs": runs,
    }


def main():
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(summary['runs'])} runs -> {SUMMARY_PATH}")
    for task, run in summary["headline"].items():
        if run:
            print(f"  {task:9s} test macro-F1 {run['macro_f1']:.3f}  (n={run['n']}, {run['file']})")
        else:
            print(f"  {task:9s} no test run matching current defaults")


if __name__ == "__main__":
    main()
