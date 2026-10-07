"""Protect evaluation denominators and the distinction between proxy and gold."""
import pytest

from src.adaptive_metrics import (aggregate_retrieval, evidence_output_metrics,
    nli_metrics, ranked_evidence_metrics, reference_region_ids, validate_gold)
from src.evidence_units import EvidenceUnit


def unit(i, text, page=1):
    return EvidenceUnit("a"*64, page, f"p{i}", "section", "de", text, (0,0,1,1), i)


def test_broad_reference_hit_cannot_create_decisive_gold():
    unrelated = unit(1, "Dieser Abschnitt beschreibt nur den allgemeinen politischen Hintergrund.")
    decisive = unit(2, "Der Bundesrat lehnt die Initiative ausdrücklich ab.")
    region = reference_region_ids(unrelated.text+"\n"+decisive.text, [unrelated, decisive])
    assert ranked_evidence_metrics([unrelated.id], [set(region)])["recall@1"] == 1
    assert ranked_evidence_metrics([unrelated.id], [{decisive.id}])["recall@1"] == 0
    assert aggregate_retrieval([None])["metrics"] is None
    assert validate_gold({"status":"provisional"}, {}, []) is None


def test_distributed_gold_requires_all_groups_and_accepts_alternatives():
    score = ranked_evidence_metrics(["b", "x", "c"], [{"a", "b"}, {"c"}])
    assert score["recall@1"] == .5
    assert score["complete@1"] is False
    assert score["complete@3"] is True
    assert score["reciprocal_rank"] == pytest.approx(1/3)
    miss = ranked_evidence_metrics(["b"], [{"a", "b"}, {"c"}])
    summary = aggregate_retrieval([score, miss])
    assert summary["metrics"]["mrr"] == pytest.approx(1/6)
    assert summary["metrics"]["mean_gold_rank_hits_only"] == 3
    assert summary["metrics"]["rank_misses"] == 1


def test_incomplete_annotations_make_precision_a_lower_bound():
    a, b = unit(1, "known", 1), unit(2, "potential alternative", 2)
    metric = evidence_output_metrics([a.id,b.id], [{a.id}], [a,b])
    assert metric["evidence_accuracy"] == .5
    assert metric["accuracy_interpretation"] == "annotation_lower_bound"
    assert metric["evidence_recall"] == 1


def test_operational_errors_are_not_neutral_or_selection_eligible():
    metrics = nli_metrics([
        {"entailment_label":0,"label":0}, {"entailment_label":1,"label":None},
        {"entailment_label":2,"label":2}])
    assert metrics["coverage"] == pytest.approx(2/3)
    assert metrics["confusion_matrix"][1] == [0,0,0]
    assert metrics["eligible_for_selection"] is False
    assert metrics["macro_f1"] == pytest.approx(2/3)
