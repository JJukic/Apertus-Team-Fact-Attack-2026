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
