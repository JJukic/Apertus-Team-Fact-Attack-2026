"""Import GPU preparation after checking sources, split and verbatim PDF units.

Only relocate booklet paths. Keep models, scores, IDs and original text intact.
Inference can then run locally without uploading API credentials to a cluster.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evidence_units import EvidenceUnitParser


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded validation/test directory")
    parser.add_argument("--output", type=Path, required=True, help="Local experiment root")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    if args.split == "test" and not (args.output/"selection.json").exists():
        raise ValueError("Freeze validation selection before importing test preparation")
    manifest = json.loads((args.source/"run.json").read_text())
    signature = manifest["signature"]
    for name, expected in signature["sources"].items():
        if digest(BASE/name) != expected:
            raise ValueError(f"Source snapshot differs: {name}")
    source = BASE/f"data/evaluation/adaptive/{args.split}.jsonl"
    if digest(source) != signature["split_sha256"]:
        raise ValueError("Dataset split differs")
    cases = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    target = args.output/args.split
    if (target/"run.json").exists() and json.loads((target/"run.json").read_text())["signature"] != signature:
        raise ValueError("Cannot replace an experiment with a different signature")
    documents, imported = {}, []
    for case in cases:
        path = args.source/"prepared"/f"{case['id']}.json"
        prepared = json.loads(path.read_text())
        pdf = BASE/f"data/booklets/{case['booklet_date']}_{case['booklet_language']}.pdf"
        sha = digest(pdf)
        if prepared["parsing"]["document_sha256"] != sha or prepared["claim"] != case["claim"]:
            raise ValueError(f"Prepared source/claim differs: {case['id']}")
        if prepared["claim_language"] != case["claim_language"] or prepared["booklet_language"] != case["booklet_language"]:
            raise ValueError(f"Prepared languages differ: {case['id']}")
        if sha not in documents:
            parser = EvidenceUnitParser()
            local_units = parser.parse(pdf, case["booklet_language"])
            if parser.VERSION != prepared["parsing"]["parser_version"]:
                raise ValueError("Parser version differs")
            local = {u.id:u for u in local_units}
            remote_path = args.source/"documents"/f"{sha}.json"
            remote = json.loads(remote_path.read_text())
            remote_units = remote["units"]
            if {u["id"] for u in remote_units} != set(local):
                raise ValueError("Cross-platform parsing produced different source IDs")
            for u in remote_units:
                original = local[u["id"]]
                if (u["text"] != original.text or u["page"] != original.page
                        or u["document_id"] != sha or u["paragraph_id"] != original.paragraph_id):
                    raise ValueError("Cross-platform source text or page differs")
            documents[sha] = remote
        by_id = {u["id"]:u for u in documents[sha]["units"]}
        for u in prepared["retrieval"]["evidence"] + [c["unit"] for c in prepared["retrieval"]["candidates"]]:
            if u["id"] not in by_id or u != by_id[u["id"]]:
                raise ValueError("Prepared evidence/candidate differs from original document units")
        prepared["preparation_origin"] = {"source_booklet":prepared["booklet"],
            "source_prepared_sha256":digest(path), "source_run_sha256":digest(args.source/"run.json")}
        prepared["booklet"] = str(pdf.resolve())
        output = target/"prepared"/path.name
        if output.exists() and json.loads(output.read_text()) != prepared:
            raise ValueError("Cannot overwrite different prepared data")
        write(output, prepared)
        imported.append({"case_id":case["id"], **prepared["preparation_origin"]})
    for sha, document in documents.items():
        write(target/"documents"/f"{sha}.json", document)
    if not (target/"frozen_sources").exists():
        shutil.copytree(args.source/"frozen_sources", target/"frozen_sources")
    write(target/"run.json", manifest)
    shutil.copyfile(args.source/"run.json", target/"preparation_run.original.json")
    write(target/"import_record.json", {"cases":imported,
        "verified_verbatim_documents":len(documents), "importer_sha256":digest(Path(__file__))})
    print(f"Imported {len(imported)} cases; verified original text and pages for {len(documents)} PDFs.")


if __name__ == "__main__":
    main()
