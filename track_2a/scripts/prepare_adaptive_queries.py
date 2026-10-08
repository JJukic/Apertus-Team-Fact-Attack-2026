"""Cache faithful multilingual queries locally before GPU-only preparation.

The compute cluster receives query artifacts, never an Apertus API key.
Gold labels and references are never sent to the translation model.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import sys
import threading

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.grounded_apertus import ApertusJSONTransport, QueryPreparer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Experiment root with frozen selection; required for test")
    args = parser.parse_args()
    if args.split == "test":
        if args.output is None or not (args.output / "selection.json").exists():
            parser.error("Freeze validation selection before preparing test queries")
        selection = json.loads((args.output / "selection.json").read_text())
        for name, expected in selection["sources"].items():
            if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source differs from frozen selection: {name}")
        if ApertusJSONTransport().model != selection["model"]:
            raise ValueError("Model differs from frozen selection")
    source = BASE / f"data/evaluation/adaptive/{args.split}.jsonl"
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["splits"][args.split]["sha256"]:
        raise ValueError("Dataset split changed")
    cases = [json.loads(line) for line in source.read_text().splitlines()]
    needed = sorted({(c["claim"], c["claim_language"]) for c in cases
                     if c["claim_language"] != c["booklet_language"]})
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    def translate(item):
        claim, language = item
        def journal(event):
            with lock, args.journal.open("a") as handle:
                handle.write(json.dumps(event, ensure_ascii=False)+"\n")
                handle.flush()
        transport = ApertusJSONTransport(journal=journal)
        result = QueryPreparer(transport, BASE / ".cache/adaptive/queries").prepare(claim, language, translate=True)
        return language, result["translation"]["cache_status"]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(translate, item) for item in needed]
        for index, future in enumerate(as_completed(jobs)):
            print(f"query {index+1}/{len(jobs)} {future.result()}", flush=True)


if __name__ == "__main__":
    main()
