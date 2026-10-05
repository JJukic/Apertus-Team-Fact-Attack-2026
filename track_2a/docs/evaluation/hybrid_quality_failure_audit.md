# Falltabelle zum Hybrid-Qualitätsvergleich

Die [CSV-Falltabelle](hybrid_quality_failure_audit.csv) enthält 200 Zeilen: 50 Fälle × zwei
Methoden × erster/warmer Durchlauf. Sie wurde aus den gespeicherten API- und Falljournalen
ohne neue API-Aufrufe erzeugt. Jede Zeile wurde gegen den unveränderten Datensatz geprüft.
Alle F1-Werte wurden aus den Einzel-Labels neu berechnet und gegen den Live-Bericht geprüft.

| Methode | Durchlauf | Rohlabel F1 | Nach Konfidenzregeln F1 | Pipeline F1 |
|---|---|---:|---:|---:|
| `hybrid` | `first_pass` | 0.8122 | 0.7888 | 0.6016 |
| `hybrid` | `warm_repeat` | 0.8122 | 0.7888 | 0.5593 |
| `hybrid_dense` | `first_pass` | 0.9044 | 0.9044 | 0.5865 |
| `hybrid_dense` | `warm_repeat` | 0.8832 | 0.8832 | 0.6106 |

## Wo korrekte Rohlabels verloren gingen

`first_loss_stage` beschreibt den ersten Schritt, nach dem ein zuvor korrektes
Label falsch wird. `raw_wrong_or_context_or_gold` ist keine Ursachenbehauptung:
Hier muss man Retrieval, Modellantwort und eventuell das Gold-Label einzeln prüfen.

| Methode | Durchlauf | Rohlabel falsch / Endlabel falsch | Arbiter-Flip | Zahlenregel-Flip | Belegregel-Flip | Nicht-neutral ohne zugeordneten Beleg |
|---|---|---:|---:|---:|---:|---:|
| `hybrid` | `first_pass` | 9 | 1 | 2 | 8 | 7 |
| `hybrid` | `warm_repeat` | 9 | 1 | 2 | 10 | 8 |
| `hybrid_dense` | `first_pass` | 5 | 0 | 1 | 14 | 11 |
| `hybrid_dense` | `warm_repeat` | 6 | 0 | 1 | 12 | 12 |

Die drei [fraglichen geerbten Gold-Labels](hybrid_quality_label_audit.json) sind in der CSV
markiert, aber für diese Primärauswertung nicht geändert. Die vorhandenen zwölf
`gold_evidence_pages` nennen jeweils eine hinreichende Referenzseite, nicht alle
gültigen Seiten. `reference_page_selected=false` beweist daher keinen Retrieval-Fehler.
Auch `mapped_evidence_count` beweist nicht, dass ein Beleg die Aussage semantisch stützt.
Nicht-neutrale Antworten ohne zugeordneten Beleg sind gesondert markiert; das
betrifft besonders Contradiction und muss manuell geprüft werden.
Die Felder `raw_evidence` und `final_evidence` erlauben die manuelle Kontrolle.

Für den ersten manuellen Durchgang gibt es eine separate
[14-Fälle-Prüfliste](hybrid_dense_first_pass_manual_review.csv). Sie enthält nur
`hybrid_dense` im ersten Durchlauf mit `evidence_guardrail_flip` und freie
Spalten für `support_on_selected_page` (ja/nein/unklar), die physische
`verified_support_page`, `failure_cause` und `review_note`. Sinnvolle Ursachenwerte
sind `quote_format`, `model_paraphrase`, `retrieval_miss`, `no_model_evidence`,
`gold_label_error` oder `other`. `gold_label_error` nur nach Kontrolle des
Original-PDFs setzen. Die Prüfliste wird von diesem Skript nicht überschrieben.

Reproduktion vom Repository-Root: `./.venv/bin/python track_2a/scripts/build_hybrid_failure_audit.py`.
Die CSV enthält Original-Claims und Modellzitate und sollte als Audit-Artefakt behandelt werden.
