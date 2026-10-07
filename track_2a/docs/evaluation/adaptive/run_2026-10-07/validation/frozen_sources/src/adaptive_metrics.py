"""Evaluation only: decisive gold evidence is distinct from reference-region hits.

No function in this module supplies labels or references to inference.
Unreviewed annotations never become gold metrics, and missing predictions never
become Neutral predictions.
"""
import hashlib
import re
import unicodedata

LABELS = (0, 1, 2)
K_VALUES = (1, 3, 5, 10, 20)


def claim_hash(claim):
    return hashlib.sha256(claim.encode()).hexdigest()


def reference_normalize(text):
    """Comparison normalization only; never mutate emitted evidence text."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", "", text.replace("\u00ad", "")).casefold()


def reference_region_ids(reference, units):
    """Conservative full-unit containment in the supplied broad reference.

    This is a localisation proxy, not decisive-evidence annotation. Short
    headings and repeated page numbers cannot establish a region match.
    """
    reference = reference_normalize(reference)
    return [u.id for u in units if u.kind != "heading"
            and len(reference_normalize(u.text)) >= 40
            and reference_normalize(u.text) in reference]


def nli_metrics(rows):
    """Fixed three-class metrics plus explicit operational coverage.

    Scores on successful predictions are diagnostic only when coverage is
    incomplete. Selection requires coverage == 1.
    """
    matrix = [[0]*3 for _ in LABELS]
    errors = 0
    for row in rows:
        gold, predicted = row["entailment_label"], row.get("label")
        if type(gold) is not int or gold not in LABELS:
            raise ValueError("Invalid gold label")
        if predicted is None:
            errors += 1
            continue
        if type(predicted) is not int or predicted not in LABELS:
            raise ValueError("Invalid predicted label")
        matrix[gold][predicted] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label])
        selected = sum(r[label] for r in matrix)
        precision = tp/selected if selected else 0.0
        recall = tp/support if support else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "support": support}
    successful = len(rows)-errors
    return {"count": len(rows), "successful_predictions": successful,
        "operational_errors": errors, "coverage": successful/len(rows) if rows else None,
        "eligible_for_selection": bool(rows) and errors == 0,
        "macro_f1": sum(c["f1"] for c in per_class.values())/3 if successful else None,
        "accuracy": sum(matrix[i][i] for i in LABELS)/successful if successful else None,
        "score_population": "successful_predictions_only",
        "confusion_matrix": matrix, "confusion_order": list(LABELS), "per_class": per_class}


def language_metrics(rows):
    groups = {"overall": rows}
    for language in ("de", "fr", "it"):
        groups[f"claim_{language}"] = [r for r in rows if r["claim_language"] == language]
        for document in ("de", "fr", "it"):
            groups[f"{language}->{document}"] = [r for r in rows
                if r["claim_language"] == language and r["booklet_language"] == document]
    groups["same_language"] = [r for r in rows if r["claim_language"] == r["booklet_language"]]
    groups["cross_language"] = [r for r in rows if r["claim_language"] != r["booklet_language"]]
    return {name:nli_metrics(values) for name,values in groups.items()}


def validate_gold(annotation, case, units):
    """Reject stale or inconsistent source IDs before scoring reviewed gold."""
    if annotation is None or annotation.get("status") != "verified_decisive":
        return None
    if annotation.get("case_id") != case["id"] or annotation.get("claim_sha256") != claim_hash(case["claim"]):
        raise ValueError("Gold annotation belongs to a different claim")
    if not units or annotation.get("document_sha256") != units[0].document_id:
        raise ValueError("Gold annotation belongs to a different PDF")
    if annotation.get("parser_version") != case["parser_version"]:
        raise ValueError("Gold annotation uses a different parser")
    if annotation.get("entailment_label") != case["entailment_label"]:
        raise ValueError("Gold label changed since evidence review")
    if case["entailment_label"] == 1:
        raise ValueError("Decisive positive/contradiction evidence is undefined for Neutral")
    groups = annotation.get("required_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Decisive gold requires at least one evidence group")
    source_ids = {u.id for u in units}
    for group in groups:
        ids = group.get("any_of")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or not set(ids) <= source_ids:
            raise ValueError("Unknown/empty/duplicate gold source IDs")
    if not annotation.get("reviewer") or not annotation.get("rationale"):
        raise ValueError("Verified gold needs review provenance and rationale")
    return [set(group["any_of"]) for group in groups]


def ranked_evidence_metrics(ranked_ids, groups, *, ks=K_VALUES):
    """Recall of required groups; each group allows equivalent source units.

    Distributed evidence requires every group. MRR uses the earliest rank where
    all requirements are available. Missing evidence contributes reciprocal rank
    zero; mean rank is explicitly conditional on a hit.
    """
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranked source IDs must be deduplicated")
    if not groups:
        return None
    ranks = {unit_id:i+1 for i,unit_id in enumerate(ranked_ids)}
    group_ranks = [min((ranks[i] for i in group if i in ranks), default=None) for group in groups]
    complete_rank = max(group_ranks) if all(r is not None for r in group_ranks) else None
    result = {"required_groups": len(groups), "group_ranks": group_ranks,
        "complete_gold_rank": complete_rank,
        "reciprocal_rank": 1/complete_rank if complete_rank else 0.0,
        "candidate_recall": sum(r is not None for r in group_ranks)/len(groups),
        "candidate_complete": complete_rank is not None}
    for k in ks:
        result[f"recall@{k}"] = sum(r is not None and r <= k for r in group_ranks)/len(groups)
        result[f"complete@{k}"] = complete_rank is not None and complete_rank <= k
    return result


def evidence_output_metrics(selected_ids, groups, units, *, exhaustive=False):
    if not groups:
        return None
    by_id = {u.id:u for u in units}
    if not set(selected_ids) <= set(by_id):
        raise ValueError("Output evidence is not from the evaluated document")
    selected = set(selected_ids)
    known = set().union(*groups)
    pages = {by_id[i].page for i in known}
    denominator = len(selected)
    return {
        "evidence_recall": sum(bool(selected & group) for group in groups)/len(groups),
        "evidence_complete": all(selected & group for group in groups),
        "evidence_accuracy": len(selected & known)/denominator if denominator else 0.0,
        "page_accuracy": sum(by_id[i].page in pages for i in selected)/denominator if denominator else 0.0,
        "accuracy_interpretation": "exact" if exhaustive else "annotation_lower_bound",
        "unannotated_output_ids": sorted(selected-known),
    }


def aggregate_retrieval(results):
    reviewed = [r for r in results if r is not None]
    if not reviewed:
        return {"measured_cases": 0, "metrics": None}
    hits = [r["complete_gold_rank"] for r in reviewed if r["complete_gold_rank"] is not None]
    keys = [f"recall@{k}" for k in K_VALUES] + [f"complete@{k}" for k in K_VALUES]
    return {"measured_cases": len(reviewed),
        "metrics": {**{key:sum(r[key] for r in reviewed)/len(reviewed) for key in keys},
            "mrr": sum(r["reciprocal_rank"] for r in reviewed)/len(reviewed),
            "mean_gold_rank_hits_only": sum(hits)/len(hits) if hits else None,
            "rank_misses": len(reviewed)-len(hits),
            "candidate_recall": sum(r["candidate_recall"] for r in reviewed)/len(reviewed),
            "candidate_complete": sum(r["candidate_complete"] for r in reviewed)/len(reviewed)}}
