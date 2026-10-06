# Iteration 4: vorbereiteter Vergleich der Beleg-Prompts

Status: **nach zusätzlicher Freigabe mit 200 Anfragen ausgeführt**.
Resultate: [Live-Bericht](evidence_prompt_dev_live_2026-10-06.md).
Der folgende Abschnitt dokumentiert den eingefrorenen Versuchsplan.

## Hypothese und eingefrorener Versuch

Der optionale Zusatz in `candidate_evidence_prompt_v1.txt` verlangt kurze,
wörtliche und sachlich hinreichende Belege sowie Konsistenz zwischen Label,
Begründung und Wahrscheinlichkeiten. Er wird ausschließlich im experimentellen
Client injiziert. Der produktive `ApertusClient` und sein Standardprompt bleiben
unverändert. Zitatabgleich, Zahlenprüfung und Entscheidungsschwellen sind in
beiden Varianten identisch; allein der Promptzusatz wird verändert.

| Parameter | Festgelegter Wert |
|---|---|
| Fälle | dieselben 50 Entwicklungsfälle, unveränderte Labels |
| Methoden | BM25 (`retrieval`) und `hybrid_dense` |
| Varianten | bisheriger Prompt und bisheriger Prompt + Zusatz |
| Antworten | eine je Fall, Methode und Prompt; keine warme Wiederholung |
| Regulärer Bedarf | 50 × 2 × 2 = 200 neue API-Anfragen |
| Modell | `swiss-ai/Apertus-v1.5-70B-thinking` |
| Plan-SHA256 | `c1af8bd6f8025d370ab817752a417c8799beedbd56a34418770a8420eb06c9e5` |
| Promptdatei-SHA256 | `72877dfda59d00f70f33a78941e680bf2434509043be74310f28059e625d9d99` |

`evidence_prompt_dev_plan_2026-10-06.json` enthält 100 vorbereitete Kontexte,
deren Quellen, PDF-Prüfsummen und Implementierungsstand. Beide Promptvarianten
nutzen je Fall und Methode denselben Snapshot. Labels und Gold-Referenzen werden
nicht an Retrieval oder Modell übergeben. Die Reihenfolge der Varianten wechselt
zwischen Fällen. Dieser Lauf misst Qualität; seine Retrieval-Zeit ist wegen der
Offline-Vorbereitung kein Laufzeitvergleich der Methoden.

Die Vorbereitung lud ausschließlich vorhandene lokale Embedding-Gewichte;
Downloads, Retrieval-Fallback und generative API-Aufrufe waren deaktiviert.
Der Plan wurde nach Fertigstellung erneut gegen Dateien und Prüfsummen validiert.

## Messung und Qualitätsentscheidung

Der Runner zeichnet jede API-Antwort und jede Vorhersage getrennt auf. Er berechnet
Macro-F1 mit `labels=[0,1,2]`, Werte nach Sprache, Konfusionsmatrizen, korrigierte
und verschlechterte Labels und IDs sämtlicher geänderter Fälle. Zusätzlich zählt
er zugeordnete Originalspans. Das ist eine Prüfung der Textzuordnung, keine
semantische Prüfung. Der Abschlussstatus bleibt deshalb
`complete_pending_evidence_review`.

Alle geänderten Fälle und akzeptierten Belege müssen anschließend auf sachliche
Unterstützung beziehungsweise Widerspruch geprüft werden. Ein F1-Gewinn mit
sachfremden Belegen besteht die Qualitätsbedingung nicht. Die drei bekannten
fraglichen Gold-Labels werden weiterhin separat behandelt und nicht stillschweigend
korrigiert. Die 50 Fälle umfassen 32 unterschiedliche Claims und 18 Übersetzungs-
varianten; sie sind Entwicklungsdaten. Eine belastbare Verbesserung erfordert
einen weiteren, vorher ungenutzten Testdatensatz und dessen getrennte Freigabe.

## Budget und Betrieb

Die frühere Freigabe von 200 Anfragen wurde
mit 200 Anfragen ausgeschöpft. Dieser Versuch benötigt
eine zusätzliche Freigabe. Ohne diese bleibt er vorbereitet. Ein Geldpreis für
den Endpoint ist in den verwendeten Projektvorgaben nicht dokumentiert; tatsächliche
Eingabe- und Ausgabetokens werden aufgezeichnet, ein CHF-Betrag wird nicht erfunden.

Jeder fehlgeschlagene Versuch zählt zur Obergrenze. Zusätzliche SDK-Retries sind
deaktiviert; die vorhandenen Client-Retries laufen durch denselben Budgetzähler.
Bei einer Obergrenze von 200 können Fehler den vollständigen Vergleich verhindern.
API-/Parse-Fehler brechen ab und werden nicht als Neutral im F1 gewertet.
Vorhandene Ergebnisdateien verhindern eine versehentliche Wiederholung.

Die vollständige bestehende Testsuite einschließlich sechs neuer Tests ist
bestanden. Die neuen Tests prüfen unveränderte Baseline-Anfragen, isolierte
Promptänderung, harte Budgetgrenze auch bei Fehlern, Kontextisolation,
serialisierbare Vorbereitung ohne Gold-Leakage, Drei-Klassen-Metriken und
Ablehnung eines falschen Planhashes vor Client-Initialisierung.

## Reproduktion

Vorbereitung aus `track_2a` (die vorhandene Plandatei wird nicht überschrieben):

```sh
../.venv/bin/python scripts/run_prompt_comparison.py prepare \
  --dataset data/evaluation/hybrid_quality_50_validation.jsonl \
  --prompt docs/evaluation/candidate_evidence_prompt_v1.txt \
  --methods retrieval hybrid_dense \
  --output docs/evaluation/evidence_prompt_dev_plan_2026-10-06.json
```

**Erst nach zusätzlicher Freigabe**, maximale Obergrenze 200:

```sh
../.venv/bin/python scripts/run_prompt_comparison.py execute \
  --plan docs/evaluation/evidence_prompt_dev_plan_2026-10-06.json \
  --approved-plan-sha256 c1af8bd6f8025d370ab817752a417c8799beedbd56a34418770a8420eb06c9e5 \
  --max-api-requests 200 \
  --output docs/evaluation/evidence_prompt_dev_live_2026-10-06.json
```

Prüfsummenänderungen an Implementierung, Datensatz, PDFs, Modellkonfiguration oder
Prompt verhindern die Ausführung des alten Plans. Weitere Änderungen benötigen
eine erneute Vorbereitung und einen neuen Planhash.
