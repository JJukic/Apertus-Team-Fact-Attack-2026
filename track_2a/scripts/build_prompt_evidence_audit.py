"""Build a source-page audit from a completed paired prompt experiment, offline."""
import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.embeddings import pdf_hash
from src.evidence_matcher import match_quote
from src.pdf_parser import PDFParser


def build(plan_path, result_path, output):
    if output.exists():
        raise ValueError("Audit already exists; preserve completed manual reviews")
    plan, result = (json.loads(p.read_text()) for p in (plan_path, result_path))
    if result["status"] != "complete_pending_evidence_review":
        raise ValueError("Only a completed comparison may be audited")
    if result["plan_sha256"] != pdf_hash(plan_path):
        raise ValueError("Result belongs to a different plan")
    expected = {(s["method"], s["record"]["id"]): s for s in plan["snapshots"]}
    actual = {(r["method"], r["id"]): r for r in result["results"]}
    if len(actual) != len(result["results"]) or set(actual) != set(expected):
        raise ValueError("Cases incomplete or duplicated")
    pages = {}
    for name, digest in plan["pdf_sha256"].items():
        if pdf_hash(Path(name)) != digest:
            raise ValueError("Source PDF changed")
        pages[name] = {p["page_number"]: p for p in PDFParser().extract_pages(name)}
    rows = []
    for key, item in expected.items():
        record, pair = item["record"], actual[key]
        changed = pair["baseline"]["label"] != pair["candidate"]["label"]
        for variant in ("baseline", "candidate"):
            pred = pair[variant]["prediction"]
            quotes = pred["evidence_sources"]
            for index, source in enumerate(quotes or [None]):
                page = source["page_number"] if source else None
                text = source["quote"] if source else ""
                original = pages[item["pdf"]].get(page)
                match = match_quote(text, [original]) if original else None
                # Empty evidence is acceptable for Neutral; not an automatic
                # semantic pass for Entailment/Contradiction.
                rows.append({"method": key[0], "case_id": key[1], "variant": variant,
                             "evidence_index": index, "claim": record["claim"],
                             "claim_language": record["claim_language"],
                             "booklet_language": record["booklet_language"],
                             "gold": record["entailment_label"], "predicted": pred["label"],
                             "label_changed": changed, "page": page, "quote": text,
                             "exact_on_attributed_page": bool(match),
                             "source_span": match[1] if match else "",
                             "reasoning": pred["reasoning"],
                             "pdf": item["pdf"], "pdf_sha256": plan["pdf_sha256"][item["pdf"]],
                             "semantic_verdict": "pending" if quotes or pred["label"] != 1 else "neutral_no_quote",
                             "review_note": ""})
    if output.exists():
        raise ValueError("Existing audit may contain manual review; never overwrite it")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} audit rows; semantic evidence validity still requires review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.plan, args.result, args.output)


if __name__ == "__main__":
    main()
