"""Ablations must use their own candidates and freeze complete validation only."""
import importlib.util
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
spec = importlib.util.spec_from_file_location("comparison",BASE/"scripts/run_adaptive_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

from src.adaptive_retrieval import AdaptiveSettings
from src.evidence_units import EvidenceUnit


def preparation():
    units = [EvidenceUnit("a"*64,1,str(i),"section","de",f"source {i}",(0,0,1,1),i) for i in range(6)]
    generators = {"bm25_original":[{"id":units[0].id,"rank":1,"score":3}],
        "bm25_translated":[{"id":units[1].id,"rank":1,"score":4}],
        "dense":[{"id":units[4].id,"rank":1,"score":.8}]}
    candidates = [{"unit":u.to_dict(),"ranks":{name:1},"scores":{name:1},
                   "reranker_score":score} for u,name,score in
                  [(units[0],"bm25_original",.1),(units[1],"bm25_translated",.2),(units[4],"dense",.9)]]
    return {"retrieval":{"generators":generators,"candidates":candidates,"reranking_ms":5}},units


def test_dense_and_bm25_do_not_borrow_other_generators():
    full,units = preparation()
    for name,index in (("E1",0),("dense",4)):
        result = comparison.derive_preparation(full,name,comparison.profile(name,AdaptiveSettings()),units)
        assert result["retrieval"]["selected_ids"] == [units[index].id]
        assert len(result["retrieval"]["candidates"]) == 1
    assert len(full["retrieval"]["candidates"]) == 3


def test_union_reranking_and_neighbors_are_separate_ablations():
    full,units = preparation()
    base = AdaptiveSettings(evidence_k=1)
    union = comparison.derive_preparation(full,"E4",comparison.profile("E4",base),units)
    reranked = comparison.derive_preparation(full,"E5",comparison.profile("E5",base),units)
    expanded = comparison.derive_preparation(full,"E6r2",comparison.profile("E6r2",base),units)
    assert union["retrieval"]["selected_ids"] == [units[0].id]
    assert reranked["retrieval"]["selected_ids"] == [units[4].id]
    assert expanded["retrieval"]["selected_ids"] == [units[4].id]
    assert expanded["retrieval"]["expanded_ids"] == [u.id for u in units[2:]]


def test_partial_scores_cannot_select_configuration():
    row = {"entailment_label":0,"label":0,"claim_language":"de","booklet_language":"de",
           "cost":{"input_tokens":12,"total_latency_ms":4}}
    assert not comparison.summaries([row],2)["nli"]["overall"]["eligible_for_selection"]


def test_streaming_run_saves_predictions_and_resumes_without_repeating_calls(tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    calls = []
    source = tmp_path / "data/evaluation/adaptive/validation.jsonl"
    source.parent.mkdir(parents=True)
    cases = [{"id": str(i), "claim": f"claim {i}", "booklet_date": "2024-11-24",
              "claim_language": "de", "booklet_language": "de", "entailment_label": i}
             for i in (0, 1)]
    source.write_text("".join(json.dumps(c)+"\n" for c in cases))
    (source.parent / "manifest.json").write_text(json.dumps({"splits": {
        "validation": {"sha256": sha256(source.read_bytes()).hexdigest()}}}))
    booklet = tmp_path / "data/booklets/2024-11-24_de.pdf"
    booklet.parent.mkdir(parents=True)
    booklet.write_bytes(b"test PDF")
    for name in ("evidence_units", "grounded_apertus", "embeddings", "config", "apertus_client"):
        path = tmp_path / "src" / f"{name}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# fixture\n")

    class Engine:
        def __init__(self, **kwargs):
            self.transport = kwargs["transport"]

        def prepare(self, claim, path, **kwargs):
            full, units = preparation()
            full["retrieval"]["evidence"] = units
            full.update(claim=claim, booklet=str(path.resolve()),
                        parsing={"document_sha256": comparison.digest(path)})
            return full

        def document(self, path, language):
            _, units = preparation()
            return units, {"document_sha256": comparison.digest(path)}

        def verify(self, claim, path, **kwargs):
            assert "entailment_label" not in kwargs["prepared"]
            calls.append((kwargs["case_id"], kwargs["experiment"]))
            return {"label": 0}

    runner = tmp_path / "scripts/run_adaptive_comparison.py"
    runner.parent.mkdir()
    runner.write_text(Path(comparison.__file__).read_text())
    monkeypatch.setattr(comparison, "__file__", str(runner))
    monkeypatch.setattr(comparison.subprocess, "check_output", lambda *a, **k: "testcommit\n")
    monkeypatch.setattr(comparison, "BASE", tmp_path)
    monkeypatch.setattr(comparison, "AdaptiveVerificationEngine", Engine)
    monkeypatch.setattr(sys, "argv", ["comparison", "run", "--output", str(tmp_path / "run"),
                                      "--profiles", "E6", "E1", "--workers", "2"])
    comparison.main()
    assert set(calls) == {(c["id"], p) for c in cases for p in ("E6", "E1")}
    comparison.main()
    assert len(calls) == 4
    for case in cases:
        row = comparison.read(tmp_path / "run/validation/predictions/E6" / f"{case['id']}.json")
        assert row["entailment_label"] == case["entailment_label"]
