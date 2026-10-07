"""Validation-only Neutral threshold search; no hand-selected production cutoffs."""
from dataclasses import dataclass, asdict
import hashlib
import json
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class NeutralPolicy:
    signal: str
    threshold: float
    validation_sha256: str
    validation_macro_f1: float
    baseline_macro_f1: float

    def apply(self, label, probabilities, signals):
        if label == 1 or self.signal == "disabled":
            return label
        value = probabilities["neutral"] if self.signal == "p_neutral" else signals[self.signal]
        neutral = value >= self.threshold if self.signal == "p_neutral" else value < self.threshold
        return 1 if neutral else label

    def to_dict(self):
        return asdict(self)


def search_neutral_policy(rows):
    """Search observed validation boundaries; ties retain the uncalibrated model.

    The same rows select thresholds and report selection performance, not a test
    estimate. Test records are explicitly rejected. Raw predictions stay saved.
    """
    if not rows or any(row["split"] not in ("dev", "validation") for row in rows):
        raise ValueError("Neutral thresholds require explicit development/validation predictions")
    gold = [r["gold"] for r in rows]
    labels = [r["label"] for r in rows]
    baseline = float(f1_score(gold, labels, labels=[0,1,2], average="macro", zero_division=0))
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    best = NeutralPolicy("disabled", 0, fingerprint, baseline, baseline)
    grid = [{**best.to_dict(), "changed_cases": 0}]
    for signal in ("p_neutral", "best_reranker_score", "reranker_margin", "retrieval_agreement"):
        values = [r["probabilities"]["neutral"] if signal == "p_neutral" else r["signals"][signal] for r in rows]
        unique = sorted(set(float(v) for v in values))
        boundaries = unique + [(a+b)/2 for a,b in zip(unique, unique[1:])]
        # Also evaluate the no-intervention endpoints. Simpler disabled wins ties.
        for threshold in sorted(set(boundaries)):
            policy = NeutralPolicy(signal, threshold, fingerprint, 0, baseline)
            predicted = [policy.apply(r["label"], r["probabilities"], r["signals"]) for r in rows]
            score = float(f1_score(gold, predicted, labels=[0,1,2], average="macro", zero_division=0))
            grid.append({"signal": signal, "threshold": threshold, "macro_f1": score,
                         "changed_cases": sum(a!=b for a,b in zip(labels,predicted))})
            if score > best.validation_macro_f1+1e-12:
                best = NeutralPolicy(signal, threshold, fingerprint, score, baseline)
    return best, grid
