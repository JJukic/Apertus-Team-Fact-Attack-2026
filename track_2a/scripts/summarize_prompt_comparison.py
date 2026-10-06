"""Reproduce paired live metrics and source audit counts without API calls."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.embeddings import pdf_hash
from run_prompt_comparison import summarize


def main():
    folder = BASE / "docs/evaluation"
    stem = "evidence_prompt_dev_live_2026-10-06"
    path = folder / f"{stem}.json"
    plan_path = folder / "evidence_prompt_dev_plan_2026-10-06.json"
    audit_path = folder / f"{stem}.audit.csv"
    result, plan = json.loads(path.read_text()), json.loads(plan_path.read_text())
    api = [json.loads(line) for line in path.with_suffix(".api.jsonl").read_text().splitlines() if line.strip()]
    if result["status"] != "complete_pending_evidence_review" or result["plan_sha256"] != pdf_hash(plan_path):
        raise ValueError("Incomplete comparison or changed plan")
    if len(api) != result["api_requests_used"] or any(e["status"] != "success" for e in api):
        raise ValueError("API journal does not match successful paired comparison")
    if any(e["model_requested"] != plan["model"] or e["response"]["model"] != plan["model"] for e in api):
        raise ValueError("Requested/returned models differ from the frozen plan")
    suspect = {r["case_id"] for r in json.loads((folder / "hybrid_quality_label_audit.json").read_text())["audit"]}
    audit = list(csv.DictReader(audit_path.open()))
    summaries = {m: summarize([r for r in result["results"] if r["method"] == m]) for m in plan["methods"]}
    if summaries != result["reports"]:
        raise ValueError("Stored metrics cannot be reproduced")
    payload = {"kind": "paired_live_prompt_comparison", "model": plan["model"],
               "api_requests": len(api), "source_sha256": {p.name: pdf_hash(p) for p in
                   [path, plan_path, audit_path, path.with_suffix(".api.jsonl"), Path(__file__)]},
               "reports": summaries,
               "excluding_suspect_gold": {m: summarize([r for r in result["results"]
                   if r["method"] == m and r["id"] not in suspect]) for m in plan["methods"]},
               "evidence_audit": {m: {v: {"quotes": len(items),
                   "exact_on_attributed_page": sum(r["exact_on_attributed_page"] == "True" for r in items),
                   "manual_verdicts": dict(Counter(r["semantic_verdict"] for r in items))}
                   for v in ("baseline", "candidate")
                   for items in [[r for r in audit if r["method"] == m and r["variant"] == v and r["quote"]]]}
                   for m in plan["methods"]},
               "quality_status": "failed_or_incomplete_not_promoted",
               "note": "Development data only; unchanged gold labels. Source text matches do not prove semantic or visible-page validity."}
    (folder / f"{stem}.summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Iteration 4: Live-Vergleich der Beleg-Prompts", "",
             f"Modell: `{plan['model']}`, durch alle angefragten und zurückgegebenen Modell-IDs bestätigt.",
             f"Genau {len(api)} transportseitig erfolgreiche API-Anfragen, kein Budgetübertritt. 50 identische Entwicklungsfälle",
             "für BM25 und hybrid_dense, jeweils einmal mit bisherigem und neuem Prompt.",
             "Retrieval-Kontext, Quellen, Labels und übrige Entscheidungsregeln je Methodenpaar eingefroren.",
             "Kein unabhängiger Test; 32 unterschiedliche Claims und 18 Übersetzungsvarianten.", "",
             "**Gültigkeitsaudit: Drei Antworten enden mit finish_reason=length.**",
             "Die Scores dokumentieren die alte Verarbeitung, keinen vollständig gültigen Vergleich.",
             "Siehe [Antwortaudit](response_validity_iteration_6.json).", "",
             "| Methode | Bisheriger Prompt | Neuer Prompt | Falsch → richtig | Richtig → falsch |",
             "|---|---:|---:|---:|---:|"]
    for m,s in summaries.items():
        lines.append(f"| {m} | {s['baseline']['macro_f1']:.4f} | {s['candidate']['macro_f1']:.4f} | {s['corrected']} | {s['regressed']} |")
    lines += ["", "## Macro-F1 nach Sprache", "", "| Methode | Sprache | Bisherig | Neu |", "|---|---|---:|---:|"]
    for m,s in summaries.items():
        for lang,values in s["by_language"].items():
            lines.append(f"| {m} | {lang} | {values['baseline']['macro_f1']:.4f} | {values['candidate']['macro_f1']:.4f} |")
    lines += ["", "## Konfusionsmatrizen", "", "Zeilen Gold, Spalten Vorhersage, Reihenfolge immer 0 / 1 / 2.", ""]
    for m,s in summaries.items():
        for variant in ("baseline", "candidate"):
            lines.append(f"- {m} / {variant}: `{s[variant]['confusion_matrix_labels_0_1_2']}`")
    lines += ["", "## Belegprüfung", "", "| Methode | Variante | Zitate | Wörtlich auf zugeordneter Textschicht-Seite |", "|---|---|---:|---:|"]
    for m,variants in payload["evidence_audit"].items():
        for v,s in variants.items():
            lines.append(f"| {m} | {v} | {s['quotes']} | {s['exact_on_attributed_page']} |")
    lines += ["", "Die Textschicht-Zuordnung ist kein Nachweis semantischer oder sichtbarer Seitenrichtigkeit.",
              "Die vollständige Audit-Tabelle enthält 234 Zeilen; noch offene Sicht-/Inhaltsprüfungen",
              "stehen ausdrücklich auf `pending`.", "",
              "## Geänderte Entscheidungen", "", "| Methode | Fall | Vorher | Neu | Gold |", "|---|---|---:|---:|---:|"]
    for row in result["results"]:
        if row["baseline"]["label"] != row["candidate"]["label"]:
            lines.append(f"| {row['method']} | {row['id']} | {row['baseline']['label']} | {row['candidate']['label']} | {row['gold']} |")
    lines += ["", "## Qualitätsentscheidung und Fehlerursachen", "",
              "**Kein bestätigter Qualitätsgewinn; Prompt nicht produktiv übernommen.**",
              "BM25 gewinnt nur `historical-11`, dessen neues Zitat keine zugeordnete Originalseite hat",
              "und eine umformulierte Tabellenbeschreibung enthält. Seine wörtliche Belegquote verschlechtert sich.",
              "Dense gewinnt vier und verliert drei Entscheidungen. Bei `historical-06` ist das neue Zitat",
              "umgestellt; bei `historical-10` verbindet es Text mit Auslassungen. Beide zuvor richtigen",
              "Entailments werden durch die Belegprüfung zu Neutral. `historical-24` verliert bereits im",
              "Modelloutput: Neutral trotz widersprechender Begründung. Keine Lockerung des Matchers als Reparatur.",
              "Die verbesserten Zitate für `historical-04`, `historical-12` und `cross-it-fr-0`",
              "sind sachlich passend. `historical-20` nutzt diesmal die passende Seite 10.",
              "Bei `historical-11` genügt die allein zitierte Mindest-/Höchstanteilsangabe nicht als",
              "formaler Ausschluss eines 100-Prozent-Kantonsanteils; die gemeinsame Finanzierung muss",
              "im Beleg selbst erkennbar sein.",
              "Zusätzlich steht der bei `cross-fr-de-2` als Seite 33 ausgegebene Mietrecht-Absatz",
              "sichtbar auf Seite 32. Die Textschicht dupliziert ihn auf Seite 33. Der Ausschnitt enthält",
              "zudem nur das schriftliche Gesuch, nicht die Zustimmung. Sichtprüfung von Seiten 32/33 dokumentiert.",
              "Die drei fraglichen übernommenen Gold-Labels wurden nicht geändert; die ergänzende",
              "JSON-Zusammenfassung berechnet die Sensitivität ohne sie separat.", "",
              "## Nächster Versuch", "",
              "Zuerst die Seitenzuordnung der PDF-Textschicht separat untersuchen und konservativ prüfen.",
              "Danach ein isoliertes Verfahren zur Auswahl vollständiger, relevanter Originalspans erproben.",
              "Keine neue Live-Schleife und kein Testset-Tuning: die zusätzlich freigegebenen 200 Anfragen",
              "sind ausgeschöpft. Ein belastbarer Gewinn erfordert weiterhin ungenutzte Testdaten.", "",
              "Reproduktion ohne API aus `track_2a`:", "", "```sh",
              "../.venv/bin/python scripts/summarize_prompt_comparison.py", "```", ""]
    (folder / f"{stem}.md").write_text("\n".join(lines))
    print("Live report, reproducible metrics and audit counts saved. Candidate not promoted.")


if __name__ == "__main__":
    main()
