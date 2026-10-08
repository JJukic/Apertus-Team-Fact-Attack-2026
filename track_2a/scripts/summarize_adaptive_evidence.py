"""Evaluate frozen validation retrieval without model calls or reindexing.

Reviewed decisive gold and broad reference regions are reported separately.
This script never supplies either annotation type to prediction.
"""
import argparse
import json
from pathlib import Path

import run_adaptive_comparison as comparison
from src.adaptive_metrics import aggregate_retrieval, ranked_evidence_metrics, reference_region_ids, validate_gold
from src.adaptive_retrieval import AdaptiveSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output / "validation"
    signature = comparison.read(folder / "run.json")["signature"]
    source = comparison.BASE / "data/evaluation/adaptive/validation.jsonl"
    if comparison.digest(source) != signature["split_sha256"]:
        raise ValueError("Validation split changed")
    for name, expected in signature["sources"].items():
        if comparison.digest(comparison.BASE / name) != expected:
            raise ValueError(f"Source changed: {name}")
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    gold_path = source.parent / "validation_gold.jsonl"
    gold = {a["case_id"]: a for a in (json.loads(line) for line in gold_path.read_text().splitlines() if line.strip())}
    settings = AdaptiveSettings(**signature["settings"])
    rows = []
    for case in cases:
        full = comparison.restore(comparison.read(folder / "prepared" / f"{case['id']}.json"))
        document = comparison.read(folder / "documents" / f"{full['parsing']['document_sha256']}.json")
        units = [comparison.unit(u) for u in document["units"]]
        groups = validate_gold(gold.get(case["id"]), {**case, "parser_version": full["parsing"]["parser_version"]}, units)
        regions = reference_region_ids(case["reference_string"], units) if case["entailment_label"] != 1 else []
        for name in signature["profiles"]:
            prepared = comparison.derive_preparation(full, name, comparison.profile(name, settings), units)
            retrieval = prepared["retrieval"]
            selected = set(retrieval["selected_ids"])
            expanded = set(retrieval["expanded_ids"])
            rows.append({"id": case["id"], "profile": name,
                "claim_language": case["claim_language"], "booklet_language": case["booklet_language"],
                "gold": ranked_evidence_metrics(retrieval["ranked_ids"], groups),
                "gold_complete_selected": all(selected & group for group in groups) if groups else None,
                "gold_complete_expanded": all(expanded & group for group in groups) if groups else None,
                "reference_region_proxy": ranked_evidence_metrics(retrieval["ranked_ids"], [set(regions)]) if regions else None})
    summary = {}
    for name in signature["profiles"]:
        selected = [r for r in rows if r["profile"] == name]
        reviewed = [r for r in selected if r["gold"] is not None]
        summary[name] = {"gold": aggregate_retrieval([r["gold"] for r in selected]),
            "complete_selected": sum(r["gold_complete_selected"] for r in reviewed)/len(reviewed) if reviewed else None,
            "complete_expanded": sum(r["gold_complete_expanded"] for r in reviewed)/len(reviewed) if reviewed else None,
            "reference_region_proxy": aggregate_retrieval([r["reference_region_proxy"] for r in selected])}
    comparison.write(folder / "retrieval_results.json", {"split": "validation", "api_requests": 0,
        "split_sha256": signature["split_sha256"], "gold_sha256": comparison.digest(gold_path),
        "evaluator_sha256": comparison.digest(Path(__file__)), "profiles": summary,
        "limitation": "Only three reviewed decisive cases. Broad reference hits are not decisive evidence recall."})
    (folder / "retrieval_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    lines = ["# Validation retrieval", "", "Only reviewed decisive gold contributes to the following table.", "",
        "| Profile | Reviewed cases | Gold recall@5 | Gold recall@20 | All groups in selected evidence | After neighbors |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, report in summary.items():
        m = report["gold"]["metrics"]
        if name == "E0" and m:
            lines.append(f"| {name} | {report['gold']['measured_cases']} | n/a | n/a | {report['complete_selected']:.3f} | {report['complete_expanded']:.3f} |")
        else:
            lines.append(f"| {name} | {report['gold']['measured_cases']} | {m['recall@5']:.3f} | {m['recall@20']:.3f} | {report['complete_selected']:.3f} | {report['complete_expanded']:.3f} |" if m else f"| {name} | 0 | — | — | — | — |")
    lines += ["", "Three reviewed cases cannot establish general retrieval quality.",
        "E0 supplies the entire document; its source order is not a retrieval ranking, so recall@k is not compared.",
        "Separate broad reference-region proxy metrics are in retrieval_results.json. No API calls were made."]
    (folder / "retrieval_results.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
