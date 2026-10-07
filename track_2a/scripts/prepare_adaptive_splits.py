"""Prepare pinned official booklet-disjoint splits without model calls."""
import hashlib
import json
import argparse
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd"
SOURCE_REVISION = "9ff08597fb79dc68cbb3af9eb1388f34d21223e6"
DATES = {"train": {"2024-06-09", "2024-09-22"}, "validation": {"2024-11-24"},
         "test": {"2025-11-30", "2026-03-08"}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-prepared", action="store_true",
                        help="Repair only a not-yet-used prepared split")
    args = parser.parse_args()
    source = BASE / ".cache/evaluation/ost_v1.1.jsonl"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Official snapshot differs from the pinned revision")
    original = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    folder = BASE / "data/evaluation/adaptive"
    folder.mkdir(parents=True, exist_ok=True)
    if args.replace_prepared:
        prior = json.loads((folder / "manifest.json").read_text())
        if prior["status"] != "prepared_no_predictions" or any(
                (BASE / "docs/evaluation/adaptive").glob("*.api.jsonl")):
            raise ValueError("Cannot replace a split after model evaluation began")
    manifest = {"status": "prepared_no_predictions", "source_revision": SOURCE_REVISION,
                "source_sha256": SOURCE_SHA256,
                "source_url": f"https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets/resolve/{SOURCE_REVISION}/v1.1.jsonl",
                "gold_evidence_note": "References are broad regions. Verified decisive-unit annotations remain required.",
                "split_policy": "Entire booklet dates held out, all three document languages together.",
                "splits": {}, "normalized_exact_claim_overlap": {}}
    signatures, split_rows = {}, {}
    for split, dates in DATES.items():
        rows = [{"id": f"ost-adaptive-{index:04d}", "split": split, "source_row_index": index,
                 "claim": record["claim"], "claim_language": record["claim_language"],
                 "booklet_language": record["reference_language"],
                 "booklet_date": record["booklet_publish_date"], "booklet_url": record["booklet_url"],
                 "vote": record["vote"], "entailment_label": record["entailment_label"],
                 "reference_string": record["reference_string"]}
                for index,record in enumerate(original) if record["booklet_publish_date"] in dates]
        split_rows[split] = rows
        signatures[split] = {" ".join(row["claim"].casefold().split()) for row in rows}
    protected = signatures["validation"] | signatures["test"]
    retained_train = [row for row in split_rows["train"] if " ".join(row["claim"].casefold().split()) not in protected]
    manifest["excluded_training_exact_claim_duplicates"] = len(split_rows["train"])-len(retained_train)
    split_rows["train"] = retained_train
    for split, rows in split_rows.items():
        dates = DATES[split]
        payload = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows)
        path = folder / f"{split}.jsonl"
        if path.exists() and path.read_text() != payload and not args.replace_prepared:
            raise ValueError("Refusing to change an existing split")
        path.write_text(payload)
        signatures[split] = {" ".join(row["claim"].casefold().split()) for row in rows}
        manifest["splits"][split] = {"file": str(path.relative_to(BASE)), "sha256": digest(path),
            "count": len(rows), "dates": sorted(dates),
            "labels": dict(Counter(row["entailment_label"] for row in rows)),
            "language_pairs": dict(Counter(f'{row["claim_language"]}->{row["booklet_language"]}' for row in rows))}
    for left,right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        manifest["normalized_exact_claim_overlap"][f"{left}->{right}"] = len(signatures[left]&signatures[right])
    # No baseline_score is copied. Gold fields stay in scoring/annotation only.
    path = folder / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest and not args.replace_prepared:
        raise ValueError("Split manifest changed; investigate before reusing the test")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"counts": {s:v["count"] for s,v in manifest["splits"].items()},
                      "exact_claim_overlap": manifest["normalized_exact_claim_overlap"],
                      "api_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
