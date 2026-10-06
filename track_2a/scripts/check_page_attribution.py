"""Offline experiment: exclude off-page origins and nonpainting text modes.

This is a coordinate diagnostic, not a full visibility/OCR implementation. It
does not change production parsing, retrieval, model outputs or final labels.
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from pypdf import PdfReader
from src.embeddings import pdf_hash
from src.evidence_matcher import match_quote


def text_origin(cm, tm):
    return (tm[4] * cm[0] + tm[5] * cm[2] + cm[4],
            tm[4] * cm[1] + tm[5] * cm[3] + cm[5])


def bounded_pages(path):
    reader = PdfReader(path)
    pages, removed = [], []
    for number, page in enumerate(reader.pages, 1):
        left, bottom, right, top = map(float, page.cropbox)
        parts = []
        rendering_mode, graphics_stack = 0, []
        def before_operand(operator, arguments, cm, tm):
            nonlocal rendering_mode
            if operator == b"q":
                graphics_stack.append(rendering_mode)
            elif operator == b"Q" and graphics_stack:
                rendering_mode = graphics_stack.pop()
            elif operator == b"Tr":
                rendering_mode = int(arguments[0])
        def visit(text, cm, tm, font, size):
            if not text.strip():
                parts.append(text)
                return
            x, y = text_origin(cm, tm)
            if left <= x <= right and bottom <= y <= top and rendering_mode not in (3, 7):
                parts.append(text)
            else:
                removed.append({"page": number, "x": x, "y": y, "text": text,
                                "rendering_mode": rendering_mode})
                # Do not silently stitch text across a discarded fragment.
                parts.append("\n<excluded-off-page-fragment>\n")
        page.extract_text(visitor_text=visit, visitor_operand_before=before_operand)
        pages.append({"page_number": number, "text": "".join(parts).strip()})
    return pages, removed


def main():
    folder = BASE / "docs/evaluation"
    stem = "evidence_prompt_dev_live_2026-10-06"
    plan_path = folder / "evidence_prompt_dev_plan_2026-10-06.json"
    result_path = folder / f"{stem}.json"
    audit_path = folder / f"{stem}.audit.csv"
    plan, result = json.loads(plan_path.read_text()), json.loads(result_path.read_text())
    if result["plan_sha256"] != pdf_hash(plan_path):
        raise ValueError("Result and plan differ")
    snapshots = {(s["method"], s["record"]["id"]): s for s in plan["snapshots"]}
    sources, removal = {}, {}
    for path, digest in plan["pdf_sha256"].items():
        if pdf_hash(Path(path)) != digest:
            raise ValueError("Original PDF changed")
        sources[path], removal[path] = bounded_pages(path)
    rows = list(csv.DictReader(audit_path.open()))
    diagnostics = []
    for row in rows:
        if not row["quote"]:
            continue
        snapshot = snapshots[row["method"], row["case_id"]]
        pages = sources[row["pdf"]]
        attributed = [p for p in pages if str(p["page_number"]) == row["page"]]
        matched = match_quote(row["quote"], attributed)
        selected = snapshot["prepared"]["metrics"]["selected_pages"]
        # Candidate alternatives are limited to the original retrieved pages.
        alternative = match_quote(row["quote"], [p for p in pages if p["page_number"] in selected])
        diagnostics.append({"method": row["method"], "case_id": row["case_id"],
                            "variant": row["variant"], "quote": row["quote"],
                            "previous_page": row["page"],
                            "previous_text_layer_match": row["exact_on_attributed_page"] == "True",
                            "bounded_attributed_page_match": bool(matched),
                            "bounded_selected_page": alternative[0]["page_number"] if alternative else None})
    payload = {"kind": "offline_coordinate_attribution_diagnostic", "api_requests": 0,
               "labels_changed": 0, "f1_note": "Labels frozen: F1 unchanged by construction, not a new pipeline score.",
               "frozen_classification_metrics": result["reports"],
               "input_sha256": {p.name: pdf_hash(p) for p in [plan_path, result_path, audit_path, Path(__file__)]},
               "removed_fragment_counts": {Path(p).name: len(v) for p,v in removal.items()},
               "affected_quotes": [r for r in diagnostics if r["previous_text_layer_match"] != r["bounded_attributed_page_match"]],
               "all_quote_diagnostics": diagnostics,
               "limits": "Checks origins and nonpainting modes Tr=3/7 only. Does not account for arbitrary clipping, opacity, occlusion or incorrect visitor coordinates. Not integrated into production."}
    output = folder / "page_attribution_diagnostic_2026-10-06.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"removed_fragments": payload["removed_fragment_counts"],
                      "affected_quotes": [{k:v for k,v in r.items() if k != "quote"}
                                          for r in payload["affected_quotes"]]}, indent=2))


if __name__ == "__main__":
    main()
