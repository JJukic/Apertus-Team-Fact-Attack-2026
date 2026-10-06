"""Build a case-level, offline audit from the completed 50-case live comparison.

No model is loaded and no Apertus request is made. The API and case journals are
paired by strategy/phase and recorded order, then checked against the dataset.
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from historical_apertus_client import ApertusClient  # noqa: E402

EVAL = BASE / "docs" / "evaluation"
DATASET = BASE / "data" / "evaluation" / "hybrid_quality_50_validation.jsonl"
STEM = "hybrid_quality_2026-10-05"
METHODS = ("hybrid", "hybrid_dense")
PHASES = ("first_pass", "warm_repeat")
FIELDS = (
    "strategy", "phase", "case_index_1based", "case_id", "source_case_index",
    "evaluation_group", "claim_language", "booklet_language", "claim",
    "gold_label_inherited", "suspect_gold_label", "gold_evidence_pages",
    "selected_pages", "reference_page_selected", "raw_label", "raw_correct_inherited_gold",
    "arbiter_label",
    "after_numerical_label", "final_label", "final_correct_inherited_gold",
    "first_loss_stage", "raw_p_entail", "raw_p_neutral", "raw_p_contra",
    "raw_evidence_count", "mapped_evidence_count", "mapped_evidence_pages",
    "nonneutral_without_mapped_evidence",
    "numerical_conflict", "final_decision_rule", "raw_evidence", "final_evidence",
    "input_tokens", "output_tokens", "pipeline_latency_ms",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_after_stages(parsed, prediction):
    raw = int(parsed.get("label", 1))
    if raw not in (0, 1, 2):
        raw = 1
    probs = tuple(float(parsed.get(name, 0.0)) for name in
                  ("p_entail", "p_neutral", "p_contra"))
    arbiter, _ = ApertusClient._apply_calibrated_decision(*probs, raw)
    after_numerical = 2 if prediction["numerical_conflict"] and arbiter != 2 else arbiter
    raw_evidence = parsed.get("evidence", [])
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    if not isinstance(raw_evidence, list):
        raise ValueError("Unexpected model evidence type")
    after_vacuity = after_numerical
    if after_vacuity == 0 and not any(isinstance(e, str) and len(e.strip()) >= 15
                                       for e in raw_evidence):
        after_vacuity = 1
    if after_vacuity == 0 and not prediction["evidence_sources"]:
        after_vacuity = 1
    if after_vacuity != prediction["label"]:
        raise ValueError("Stored final label cannot be reconstructed from recorded stages")
    return raw, arbiter, after_numerical, probs, raw_evidence


def first_loss(gold, raw, arbiter, after_numerical, final):
    if final == gold:
        return "correct_final_raw_repaired" if raw != gold else "correct_throughout"
    if raw != gold:
        return "raw_wrong_or_context_or_gold"
    if arbiter != gold:
        return "arbiter_flip"
    if after_numerical != gold:
        return "numerical_rule_flip"
    return "evidence_guardrail_flip"


def main():
    report_path = EVAL / f"{STEM}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = read_jsonl(DATASET)
    audit = json.loads((EVAL / "hybrid_quality_label_audit.json").read_text(encoding="utf-8"))
    suspect = {entry["case_id"] for entry in audit["audit"]}
    if report["status"] != "complete" or report["mock_mode"] or report["dataset_sha256"] != digest(DATASET):
        raise ValueError("The completed live report and current dataset do not match")
    if any(digest(BASE / "src" / name) != expected
           for name, expected in report["source_sha256"].items()):
        raise ValueError("Current source code differs from the code used for the live report")
    if len(dataset) != 50 or report["api_requests_used"] != 200:
        raise ValueError("Expected the completed 50-case, four-phase comparison")

    api_groups = defaultdict(list)
    case_groups = defaultdict(list)
    for event in read_jsonl(EVAL / f"{STEM}.api.jsonl"):
        api_groups[(event["strategy"], event["phase"])].append(event)
    for event in read_jsonl(EVAL / f"{STEM}.cases.jsonl"):
        case_groups[(event["strategy"], event["phase"])].append(event)
    expected_groups = {(method, phase) for method in METHODS for phase in PHASES}
    if set(api_groups) != expected_groups or set(case_groups) != expected_groups:
        raise ValueError("Unexpected or missing method/pass group")

    rows = []
    summaries = []
    for method in METHODS:
        for phase in PHASES:
            key = (method, phase)
            api_events = api_groups[key]
            case_events = case_groups[key]
            if len(api_events) != len(case_events) or len(case_events) != len(dataset):
                raise ValueError(f"Incomplete journal group: {key}")
            if [x["case_index"] for x in case_events] != list(range(len(dataset))):
                raise ValueError(f"Cases are out of order: {key}")
            labels = {"gold": [], "raw": [], "arbiter": [], "final": []}
            origins = Counter()
            for i, (record, api, case) in enumerate(zip(dataset, api_events, case_events)):
                if api["status"] != "success" or case["dataset_record"] != record:
                    raise ValueError(f"Unmatched or failed API/case entry: {key}, {i}")
                prediction = case["prediction"]
                if prediction["requested_strategy"] != method or prediction["strategy"] != method:
                    raise ValueError(f"Unexpected actual strategy: {key}, {i}")
                content = api["response"]["choices"][0]["message"]["content"]
                parsed = ApertusClient._parse_json(content)
                raw, arbiter, after_numerical, probs, raw_evidence = label_after_stages(parsed, prediction)
                gold = int(record["entailment_label"])
                final = int(prediction["label"])
                origin = first_loss(gold, raw, arbiter, after_numerical, final)
                origins[origin] += 1
                for name, value in (("gold", gold), ("raw", raw), ("arbiter", arbiter), ("final", final)):
                    labels[name].append(value)
                gold_pages = record.get("gold_evidence_pages") or []
                selected = prediction["retrieval_metrics"]["selected_pages"]
                mapped = prediction["evidence_sources"]
                rows.append({
                    "strategy": method, "phase": phase, "case_index_1based": i + 1,
                    "case_id": record["id"], "source_case_index": record.get("source_case_index", ""),
                    "evaluation_group": record.get("evaluation_group", ""),
                    "claim_language": record["claim_language"],
                    "booklet_language": record["booklet_language"], "claim": record["claim"],
                    "gold_label_inherited": gold, "suspect_gold_label": record["id"] in suspect,
                    "gold_evidence_pages": json.dumps(gold_pages),
                    "selected_pages": json.dumps(selected),
                    "reference_page_selected": (bool(set(gold_pages) & set(selected)) if gold_pages else ""),
                    "raw_label": raw, "raw_correct_inherited_gold": raw == gold,
                    "arbiter_label": arbiter,
                    "after_numerical_label": after_numerical, "final_label": final,
                    "final_correct_inherited_gold": final == gold,
                    "first_loss_stage": origin,
                    "raw_p_entail": probs[0], "raw_p_neutral": probs[1], "raw_p_contra": probs[2],
                    "raw_evidence_count": len(raw_evidence), "mapped_evidence_count": len(mapped),
                    "mapped_evidence_pages": json.dumps([x.get("page_number") for x in mapped]),
                    "nonneutral_without_mapped_evidence": final != 1 and not mapped,
                    "numerical_conflict": prediction["numerical_conflict"] or "",
                    "final_decision_rule": prediction["decision_rule"] or "",
                    "raw_evidence": json.dumps(raw_evidence, ensure_ascii=False),
                    "final_evidence": json.dumps(prediction["evidence"], ensure_ascii=False),
                    "input_tokens": prediction["tokens_prompt"],
                    "output_tokens": prediction["tokens_completion"],
                    "pipeline_latency_ms": prediction["total_latency_ms"],
                })
            scores = {name: round(f1_score(labels["gold"], labels[name], average="macro",
                                           labels=[0, 1, 2], zero_division=0), 4)
                      for name in ("raw", "arbiter", "final")}
            expected_f1 = report["reports"][method][phase]["macro_f1"]
            if scores["final"] != expected_f1:
                raise ValueError(f"F1 differs from the original report: {key}")
            without_mapped = sum(row["nonneutral_without_mapped_evidence"]
                                 for row in rows[-len(dataset):])
            summaries.append((method, phase, scores, origins, without_mapped))

    csv_path = EVAL / "hybrid_quality_failure_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Falltabelle zum Hybrid-Qualitätsvergleich",
        "",
        "Die [CSV-Falltabelle](hybrid_quality_failure_audit.csv) enthält 200 Zeilen: 50 Fälle × zwei",
        "Methoden × erster/warmer Durchlauf. Sie wurde aus den gespeicherten API- und Falljournalen",
        "ohne neue API-Aufrufe erzeugt. Jede Zeile wurde gegen den unveränderten Datensatz geprüft.",
        "Alle F1-Werte wurden aus den Einzel-Labels neu berechnet und gegen den Live-Bericht geprüft.",
        "",
        "| Methode | Durchlauf | Rohlabel F1 | Nach Konfidenzregeln F1 | Pipeline F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for method, phase, scores, _, _ in summaries:
        lines.append(f"| `{method}` | `{phase}` | {scores['raw']:.4f} | {scores['arbiter']:.4f} | {scores['final']:.4f} |")
    lines += [
        "",
        "## Wo korrekte Rohlabels verloren gingen",
        "",
        "`first_loss_stage` beschreibt den ersten Schritt, nach dem ein zuvor korrektes",
        "Label falsch wird. `raw_wrong_or_context_or_gold` ist keine Ursachenbehauptung:",
        "Hier muss man Retrieval, Modellantwort und eventuell das Gold-Label einzeln prüfen.",
        "",
        "| Methode | Durchlauf | Rohlabel falsch / Endlabel falsch | Arbiter-Flip | Zahlenregel-Flip | Belegregel-Flip | Nicht-neutral ohne zugeordneten Beleg |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method, phase, _, origins, without_mapped in summaries:
        lines.append(
            f"| `{method}` | `{phase}` | {origins['raw_wrong_or_context_or_gold']} | "
            f"{origins['arbiter_flip']} | {origins['numerical_rule_flip']} | "
            f"{origins['evidence_guardrail_flip']} | {without_mapped} |"
        )
    lines += [
        "",
        "Die drei [fraglichen geerbten Gold-Labels](hybrid_quality_label_audit.json) sind in der CSV",
        "markiert, aber für diese Primärauswertung nicht geändert. Die vorhandenen zwölf",
        "`gold_evidence_pages` nennen jeweils eine hinreichende Referenzseite, nicht alle",
        "gültigen Seiten. `reference_page_selected=false` beweist daher keinen Retrieval-Fehler.",
        "Auch `mapped_evidence_count` beweist nicht, dass ein Beleg die Aussage semantisch stützt.",
        "Nicht-neutrale Antworten ohne zugeordneten Beleg sind gesondert markiert; das",
        "betrifft besonders Contradiction und muss manuell geprüft werden.",
        "Die Felder `raw_evidence` und `final_evidence` erlauben die manuelle Kontrolle.",
        "",
        "Für den ersten manuellen Durchgang gibt es eine separate",
        "[14-Fälle-Prüfliste](hybrid_dense_first_pass_manual_review.csv). Sie enthält nur",
        "`hybrid_dense` im ersten Durchlauf mit `evidence_guardrail_flip` und freie",
        "Spalten für `support_on_selected_page` (ja/nein/unklar), die physische",
        "`verified_support_page`, `failure_cause` und `review_note`. Sinnvolle Ursachenwerte",
        "sind `quote_format`, `model_paraphrase`, `retrieval_miss`, `no_model_evidence`,",
        "`gold_label_error` oder `other`. `gold_label_error` nur nach Kontrolle des",
        "Original-PDFs setzen. Die Prüfliste wird von diesem Skript nicht überschrieben.",
        "",
        "Reproduktion vom Repository-Root: `./.venv/bin/python track_2a/scripts/build_hybrid_failure_audit.py`.",
        "Die CSV enthält Original-Claims und Modellzitate und sollte als Audit-Artefakt behandelt werden.",
        "",
    ]
    (EVAL / "hybrid_quality_failure_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
