"""Imported GPU results must retain the exact local PDF evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("preparation_import", BASE / "scripts/import_adaptive_preparation.py")
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
from src.evidence_units import EvidenceUnit


@pytest.mark.parametrize("tamper", [False, True])
def test_import_checks_verbatim_evidence_and_relocates_only_path(tmp_path, monkeypatch, tamper):
    repo, remote, output = tmp_path / "repo", tmp_path / "remote", tmp_path / "output"
    case = {"id": "one", "claim": "claim", "booklet_date": "2024-11-24",
            "claim_language": "de", "booklet_language": "de"}
    dataset = repo / "data/evaluation/adaptive/validation.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps(case) + "\n")
    pdf = repo / "data/booklets/2024-11-24_de.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"fixture PDF")
    sha = importer.digest(pdf)
    unit = EvidenceUnit(sha, 1, "paragraph", "section", "de", "original text", (0, 0, 1, 1), 0)
    record = unit.to_dict()
    remote.mkdir()
    (remote / "frozen_sources").mkdir()
    manifest = {"signature": {"sources": {}, "split_sha256": importer.digest(dataset)}}
    importer.write(remote / "run.json", manifest)
    importer.write(remote / "documents" / f"{sha}.json", {"units": [record]})
    prepared = {"claim": case["claim"], "claim_language": "de", "booklet_language": "de",
                "booklet": "/cluster/booklet.pdf", "parsing": {"document_sha256": sha,
                "parser_version": "fixture"}, "retrieval": {"evidence": [dict(record)],
                "candidates": [{"unit": dict(record), "reranker_score": .8}]}}
    if tamper:
        prepared["retrieval"]["evidence"][0]["text"] = "invented evidence"
    importer.write(remote / "prepared/one.json", prepared)

    class Parser:
        VERSION = "fixture"

        def parse(self, path, language):
            assert path == pdf and language == "de"
            return [unit]

    monkeypatch.setattr(importer, "BASE", repo)
    monkeypatch.setattr(importer, "EvidenceUnitParser", Parser)
    monkeypatch.setattr(sys, "argv", ["import", "--source", str(remote), "--output", str(output)])
    if tamper:
        with pytest.raises(ValueError, match="differs from original"):
            importer.main()
        assert not (output / "validation/run.json").exists()
    else:
        importer.main()
        actual = json.loads((output / "validation/prepared/one.json").read_text())
        assert actual["booklet"] == str(pdf.resolve())
        assert actual["retrieval"] == json.loads(json.dumps(prepared["retrieval"]))
        assert actual["preparation_origin"]["source_prepared_sha256"] == importer.digest(remote / "prepared/one.json")
        assert json.loads((output / "validation/run.json").read_text()) == manifest
