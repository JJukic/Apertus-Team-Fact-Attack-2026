"""Measure original BM25 and reference alignment without any LLM calls.

Official references are broad regions. Only separately reviewed decisive-unit
annotations contribute to gold metrics. Test data is deliberately not accepted.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.adaptive_metrics import (aggregate_retrieval, ranked_evidence_metrics,
    reference_region_ids, validate_gold)
from src.adaptive_retrieval import BM25Generator
from src.evidence_units import EvidenceUnitParser


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=.75)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = BASE / f"data/evaluation/adaptive/{args.split}.jsonl"
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["splits"][args.split]["sha256"]:
        raise ValueError("Frozen split hash changed")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    gold = {}
    if args.gold:
        for line in args.gold.read_text().splitlines():
            annotation = json.loads(line)
            if annotation["case_id"] in gold:
                raise ValueError("Duplicate gold annotation")
            gold[annotation["case_id"]] = annotation
    documents, records = {}, []
    for case in rows:
        key = (case["booklet_date"], case["booklet_language"])
        if key not in documents:
            path = BASE / f"data/booklets/{key[0]}_{key[1]}.pdf"
            extractor = EvidenceUnitParser()
            with redirect_stdout(sys.stderr):
                units = extractor.parse(path, key[1])
            bm25 = BM25Generator(units, k1=args.k1, b=args.b)
            documents[key] = units, extractor.diagnostics, bm25
            write(args.output / f"{key[0]}_{key[1]}.units.json",
                  {"parsing":extractor.diagnostics,"units":[u.to_dict() for u in units]})
        units, parsing, bm25 = documents[key]
        regions = reference_region_ids(case["reference_string"], units)
        ranked = [u.id for u,score in bm25.retrieve(case["claim"], 20)]
        groups = validate_gold(gold.get(case["id"]),
                              {**case,"parser_version":parsing["parser_version"]}, units)
        record = {"id":case["id"], "split":args.split, "claim":case["claim"],
            "entailment_label":case["entailment_label"], "claim_language":case["claim_language"],
            "booklet_language":case["booklet_language"], "booklet_date":case["booklet_date"],
            "parsing":parsing, "ranked_ids":ranked,
            "reference_region_ids":regions,
            "reference_region_pages":sorted({u.page for u in units if u.id in regions}),
            "reference_region_proxy":ranked_evidence_metrics(ranked, [set(regions)]) if regions else None,
            "gold_evidence":ranked_evidence_metrics(ranked, groups)}
        records.append(record)
    (args.output / "reference_audit.cases.jsonl").write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    pairs = Counter(f"{r['claim_language']}->{r['booklet_language']}" for r in records)
    nonneutral = [r for r in records if r["entailment_label"] != 1]
    def metrics(subset):
        eligible = [r for r in subset if r["entailment_label"] != 1]
        return {"count":len(subset), "nonneutral_count":len(eligible),
            "gold_evidence":aggregate_retrieval([r["gold_evidence"] for r in eligible]),
            "reference_region_proxy":aggregate_retrieval([r["reference_region_proxy"] for r in eligible])}
    summary = {"status":"retrieval_only", "split":args.split,
        "split_sha256":manifest["splits"][args.split]["sha256"],
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=BASE,text=True).strip(),
        "source_sha256":{str(Path(__file__).relative_to(BASE)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **{f"src/{name}.py":hashlib.sha256((BASE/f"src/{name}.py").read_bytes()).hexdigest()
               for name in ("evidence_units","adaptive_retrieval","adaptive_metrics")}},
        "configuration":{"method":"bm25_original","k1":args.k1,"b":args.b,"top_k":20},
        "api_requests":0, "nli_macro_f1":None,
        "reference_proxy_warning":"Any matching unit in a broad region; does not establish decisive evidence recall.",
        "overall":metrics(records),
        "reference_alignment_missing_ids":[r["id"] for r in records if not r["reference_region_ids"]],
        "language_pairs":{pair:metrics([r for r in records
            if f"{r['claim_language']}->{r['booklet_language']}"==pair]) for pair in sorted(pairs)},
        "same_language":metrics([r for r in records if r["claim_language"]==r["booklet_language"]]),
        "cross_language":metrics([r for r in records if r["claim_language"]!=r["booklet_language"]])}
    write(args.output / "reference_audit.json", summary)
    print(json.dumps({"cases":len(records),"nonneutral":len(nonneutral),
        "aligned_references":len(records)-len(summary["reference_alignment_missing_ids"]),
        "verified_gold_cases":summary["overall"]["gold_evidence"]["measured_cases"],
        "reference_region_proxy":summary["overall"]["reference_region_proxy"],
        "api_requests":0},indent=2))


if __name__ == "__main__":
    main()
