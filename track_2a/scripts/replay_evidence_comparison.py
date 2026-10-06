"""Replay only evidence validation using recorded API answers; no API calls.

Run from track_2a: ../.venv/bin/python scripts/replay_evidence_comparison.py
Upstream model, retrieval and numerical decisions remain frozen.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from build_hybrid_failure_audit import digest, label_after_stages, read_jsonl
from historical_apertus_client import ApertusClient, SOURCE as HISTORICAL_CLIENT
from src.evidence_matcher import match_quote
from src.numerical_checker import detect_numerical_conflict
from src.pdf_parser import PDFParser

EVAL = BASE / "docs/evaluation"
STEM = "hybrid_quality_2026-10-05"


def score(rows, column):
    return round(f1_score([r["gold"] for r in rows], [r[column] for r in rows],
                         labels=[0, 1, 2], average="macro", zero_division=0), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v2",
                        help="v1: normalized spans; v2: cited numeric check; v3: experiment requiring all quotes")
    version = parser.parse_args().version
    output = EVAL / ("evidence_replay_2026-10-06" if version == "v1"
                     else f"evidence_replay_2026-10-06_{version}")
    report = json.loads((EVAL / f"{STEM}.json").read_text())
    dataset_path = BASE / "data/evaluation/hybrid_quality_50_validation.jsonl"
    dataset = read_jsonl(dataset_path)
    if report["status"] != "complete" or report["mock_mode"] or digest(dataset_path) != report["dataset_sha256"]:
        raise ValueError("Dataset or original report is not the completed live baseline")
    # Replay the exact archived response parser/arbiter; production error
    # handling may evolve without changing these historical decisions.
    for name, expected in report["source_sha256"].items():
        source = HISTORICAL_CLIENT if name == "apertus_client.py" else BASE / "src" / name
        if name != "inference.py" and digest(source) != expected:
            raise ValueError(f"Upstream implementation differs from baseline: {name}")
    suspect = {r["case_id"] for r in json.loads((EVAL / "hybrid_quality_label_audit.json").read_text())["audit"]}
    api_groups, case_groups = defaultdict(list), defaultdict(list)
    for event in read_jsonl(EVAL / f"{STEM}.api.jsonl"):
        api_groups[(event["strategy"], event["phase"])].append(event)
    for event in read_jsonl(EVAL / f"{STEM}.cases.jsonl"):
        case_groups[(event["strategy"], event["phase"])].append(event)
    expected_groups = {(m, p) for m in ("hybrid", "hybrid_dense") for p in ("first_pass", "warm_repeat")}
    if set(api_groups) != expected_groups or set(case_groups) != expected_groups:
        raise ValueError("Incomplete comparison groups")
    pages, results, summaries = {}, [], []
    for method, phase in sorted(expected_groups):
        apis, cases = api_groups[(method, phase)], case_groups[(method, phase)]
        if len(apis) != len(dataset) or len(cases) != len(dataset):
            raise ValueError("Incomplete journals")
        rows = []
        for index, (record, api, case) in enumerate(zip(dataset, apis, cases)):
            if case["case_index"] != index or case["dataset_record"] != record or api["status"] != "success":
                raise ValueError("Journal alignment failed")
            prediction = case["prediction"]
            if prediction["strategy"] != method or prediction["requested_strategy"] != method:
                raise ValueError("Unexpected strategy fallback")
            parsed = ApertusClient._parse_json(api["response"]["choices"][0]["message"]["content"])
            _, _, before_evidence, _, evidence = label_after_stages(parsed, prediction)
            path = BASE / "data/booklets" / f'{record["booklet_date"]}_{record["booklet_language"]}.pdf'
            if path not in pages:
                if digest(path) != report["booklet_sha256"][record["booklet_language"]]:
                    raise ValueError(f"PDF differs from baseline: {path}")
                pages[path] = {p["page_number"]: p for p in PDFParser().extract_pages(path)}
            selected = prediction["retrieval_metrics"]["selected_pages"]
            sources = [pages[path][p] for p in selected]
            matches = []
            new_label = before_evidence
            if new_label == 0 and not any(len(e.strip()) >= 15 for e in evidence):
                new_label = 1
            if new_label != 1:
                for quote in evidence:
                    match = match_quote(quote, sources)
                    if match:
                        source, span = match
                        matches.append({"page": source["page_number"], "quote": span})
            if new_label == 0 and not matches:
                new_label = 1
            source_conflict = None
            if version in ("v2", "v3") and new_label == 0:
                for match in matches:
                    source_conflict = detect_numerical_conflict(record["claim"], match["quote"])
                    if source_conflict:
                        new_label = 2
                        break
            unmatched = len(evidence) - len(matches)
            complete_set_gate = version == "v3" and new_label == 0 and unmatched > 0
            if complete_set_gate:
                new_label = 1
                matches = []
            rows.append({"strategy": method, "phase": phase, "case_id": record["id"],
                         "claim": record["claim"], "language": record["claim_language"],
                         "group": record["evaluation_group"], "gold": record["entailment_label"],
                         "suspect_gold": record["id"] in suspect,
                         "old_label": prediction["label"], "new_label": new_label,
                         "changed": prediction["label"] != new_label,
                         "old_evidence_count": len(prediction["evidence_sources"]),
                         "model_evidence_count": len(evidence),
                         "unmatched_model_evidence_count": unmatched,
                         "complete_set_gate": complete_set_gate,
                         "new_evidence_count": len(matches), "new_evidence": matches,
                         "source_numerical_conflict": source_conflict.explanation if source_conflict else ""})
        old_f1, new_f1 = score(rows, "old_label"), score(rows, "new_label")
        if old_f1 != report["reports"][method][phase]["macro_f1"]:
            raise ValueError("Baseline F1 does not match original report")
        clean = [r for r in rows if not r["suspect_gold"]]
        summaries.append({"strategy": method, "phase": phase, "count": len(rows),
                          "old_macro_f1": old_f1, "new_macro_f1": new_f1,
                          "corrected": sum(r["old_label"] != r["gold"] == r["new_label"] for r in rows),
                          "regressed": sum(r["old_label"] == r["gold"] != r["new_label"] for r in rows),
                          "old_confusion_matrix_labels_0_1_2": confusion_matrix(
                              [r["gold"] for r in rows], [r["old_label"] for r in rows], labels=[0, 1, 2]).tolist(),
                          "new_confusion_matrix_labels_0_1_2": confusion_matrix(
                              [r["gold"] for r in rows], [r["new_label"] for r in rows], labels=[0, 1, 2]).tolist(),
                          "evidence_rows_with_source": sum(bool(r["new_evidence_count"]) for r in rows),
                          "model_evidence_items": sum(r["model_evidence_count"] for r in rows),
                          "matched_source_spans": sum(r["new_evidence_count"] for r in rows),
                          "excluding_suspect": {"count": len(clean), "old_macro_f1": score(clean, "old_label"),
                                                "new_macro_f1": score(clean, "new_label")},
                          "by_language": {lang: {"count": sum(r["language"] == lang for r in rows),
                              "old_macro_f1": score([r for r in rows if r["language"] == lang], "old_label"),
                              "new_macro_f1": score([r for r in rows if r["language"] == lang], "new_label")}
                              for lang in sorted({r["language"] for r in rows})}})
        results.extend(rows)
    payload = {"kind": "offline_evidence_replay", "api_requests": 0,
               "note": "Same recorded answers and selected pages; inherited labels; no independent validation or semantic evidence check.",
               "input_sha256": {p.name: digest(p) for p in
                   [dataset_path, EVAL / f"{STEM}.json", EVAL / f"{STEM}.api.jsonl", EVAL / f"{STEM}.cases.jsonl",
                    BASE / "src/evidence_matcher.py", BASE / "src/inference.py",
                    BASE / "src/numerical_checker.py", Path(__file__)]},
               "summaries": summaries, "results": results}
    output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows({**r, "new_evidence": json.dumps(r["new_evidence"], ensure_ascii=False)} for r in results)
    lines = ["# Offline-Nachmessung der Belegprüfung", "",
             "Gespeicherte Antworten des Live-Vergleichs vom 5. Oktober 2026; keine neuen API-Anfragen.",
             "Je 50 identische Fälle pro Methode und Durchlauf. Neu berechnet werden Zitatzuordnung" +
             (" und erkannte Zahlenkonflikte direkt im zugeordneten Originalzitat." if version != "v1" else "."), "",
             "| Methode | Durchlauf | Macro-F1 vorher | Macro-F1 Replay | Falsch → richtig | Richtig → falsch |",
             "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f'| {s["strategy"]} | {s["phase"]} | {s["old_macro_f1"]:.4f} | {s["new_macro_f1"]:.4f} | {s["corrected"]} | {s["regressed"]} |')
    lines += ["", "## Macro-F1 nach Sprache", "", "| Methode | Durchlauf | DE | FR | IT |", "|---|---|---:|---:|---:|"]
    for s in summaries:
        langs = s["by_language"]
        lines.append(f'| {s["strategy"]} | {s["phase"]} | {langs.get("de", {}).get("new_macro_f1", "–")} | {langs.get("fr", {}).get("new_macro_f1", "–")} | {langs.get("it", {}).get("new_macro_f1", "–")} |')
    lines += ["", "## Konfusionsmatrizen nach Replay", "",
              "Zeilen = Gold-Label, Spalten = vorhergesagtes Label; Reihenfolge 0, 1, 2.", ""]
    for s in summaries:
        lines.append(f'**{s["strategy"]} / {s["phase"]}**')
        lines += ["", "| Gold \\ Vorhersage | 0 | 1 | 2 |", "|---|---:|---:|---:|"]
        for label, row in enumerate(s["new_confusion_matrix_labels_0_1_2"]):
            lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")
        lines.append("")
    lines += ["## Originalzitat-Zuordnung", "",
              "| Methode | Durchlauf | Modellzitate | Zugeordnete Originalspans | Fälle mit mindestens einem Span |",
              "|---|---|---:|---:|---:|"]
    for s in summaries:
        lines.append(f'| {s["strategy"]} | {s["phase"]} | {s["model_evidence_items"]} | {s["matched_source_spans"]} | {s["evidence_rows_with_source"]}/50 |')
    lines += ["", "## Grenzen", "",
              "Dies ist eine kontrollierte Nachberechnung, kein neuer Live-Benchmark und kein unabhängiger Test.",
              "Die 14 manuell untersuchten Fälle wurden zur Entwicklung verwendet und sind Teil dieser 50 Fälle.",
              "Die drei fraglichen Gold-Labels bleiben unverändert; die JSON-Datei enthält zusätzlich Werte ohne diese Fälle.",
              ("V3 verwirft historical-20 wegen einer Teilzuordnung. Vollständige Zitatzuordnung allein beweist ebenfalls keine inhaltliche Unterstützung."
               if version == "v3" else "Bei historical-20 wird ein echtes, aber sachfremdes Zitat von Seite 48 akzeptiert; dies verletzt die Qualitätsbedingung."),
              "Die passende Aussage steht auf der ausgewählten Seite 10; das Modell paraphrasiert sie,",
              "weshalb sie nicht wörtlich zugeordnet wird. F1 ist daher nur ein technischer Replay-Befund, kein bestätigter",
              "Qualitätsgewinn. Exakte Zitatzuordnung ist automatisch geprüft; semantische Beleggültigkeit nicht für",
              "den gesamten Testsatz.",
              "Die zwei Durchläufe sind wiederholte Modellantworten auf dieselben Fälle, keine 100 unabhängigen Fälle.",
              "BM25/full sind in diesen gespeicherten Vergleichsjournalen nicht enthalten.", "",
              "## Geänderte Labels", "", "| Methode | Durchlauf | Fall | Vorher | Replay | Gold |",
              "|---|---|---|---:|---:|---:|"]
    for r in results:
        if r["changed"]:
            lines.append(f'| {r["strategy"]} | {r["phase"]} | {r["case_id"]} | {r["old_label"]} | {r["new_label"]} | {r["gold"]} |')
    lines += ["", "Reproduktion aus `track_2a`:", "", "```sh",
              f"../.venv/bin/python scripts/replay_evidence_comparison.py --version {version}", "```", ""]
    if version in ("v2", "v3"):
        lines[-1:-1] = ["", "Zusätzlicher Prüfschritt: erkannte Zahlenkonflikte innerhalb eines tatsächlich",
                        "zugeordneten Zitats ändern Entailment zu Contradiction. Dieser Schritt fand im Replay",
                        "nur bei `historical-11` einen Konflikt. Er prüft keine nichtnumerischen Widersprüche.", ""]
    if version == "v3":
        lines += ["", "## Experimentelle Regel", "",
                  "V3 verlangt für Entailment die Zuordnung sämtlicher Modellzitate.",
                  "Die Regel ist ausschließlich im Offline-Replay implementiert, nicht im produktiven `verify_claim`.",
                  "Sie verhindert Teilzuordnungen, ersetzt aber keine semantische Prüfung.", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
