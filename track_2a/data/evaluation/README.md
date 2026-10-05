# Explorative Qualitätsprüfung vom 5. Oktober 2026

`hybrid_quality_50_validation.jsonl` enthält 32 unveränderte Claims und Labels aus
`../benchmark_2024-11-24.jsonl` und 18 zusätzliche Sprachpaar-Varianten. Die Auswahl
der Varianten erfolgte vor dem ersten API-Aufruf und steht in der Metadatendatei.
Alle sechs unterschiedlichen Sprachpaare enthalten je einen Fall pro NLI-Klasse.

Dies ist ein historischer Validierungs-/Regressionstest mit insgesamt **32
verschiedenen Claims**, kein unabhängig annotierter Test mit 50 verschiedenen
Claims. Der offizielle Juni-Testdatensatz wird nicht verwendet. Die Konfiguration
bleibt bei beiden Methoden unverändert: zehn Seiten, 40 Kandidaten pro Kanal,
RRF-Konstante 60, BGE-M3 mit der festgelegten Modellrevision.

Die Ausgangsdaten enthalten keine verlässlichen Vorlagentitel. Beide Methoden
erhalten deshalb `vote=None`; dieser Lauf untersucht nicht den Titelterm mit
doppelter BM25-Gewichtung.

Für zwölf sprachübergreifende Entailment-/Contradiction-Fälle wurde jeweils eine
hinreichende physische Referenzseite geprüft: Seite 4 für sechs Autobahnabschnitte,
Seite 6 für das Erfordernis schriftlicher Zustimmung zur Untervermietung, in allen
drei Büchleinsprachen. Die Referenzseiten sind **keine vollständige Liste aller
zulässigen Belegseiten**. Ihr Recall misst die Abdeckung genau dieser Seiten.
Labels, Referenztext und Gold-Seiten werden ausschließlich für die Auswertung
verwendet. Sie gelangen nicht in Retrieval oder NLI-Eingaben.

Während des Laufs wurden drei fragliche übernommene Labels gefunden: Die Claims
mit 5,3 Milliarden Franken sind als Entailment annotiert, die Büchlein nennen
jedoch 4,9 Milliarden. Die ursprünglichen Labels wurden für den Lauf nicht
verändert. Die Prüfung ist in
`../../docs/evaluation/hybrid_quality_label_audit.json` dokumentiert. Ergänzende
Auswertungen müssen diese Fälle ausdrücklich kennzeichnen.

Der ausführbare Live-Lauf nutzt den vorhandenen Evaluator und die vorhandene
Inference-Pipeline. Er ergänzt ein Anfragebudget, deaktiviert zusätzliche
SDK-Retries und sichert jede Antwort und jedes Pipeline-Ergebnis in JSONL.
Auch fehlgeschlagene API-Versuche zählen zur Obergrenze. Ein API-/Parse-Fehler
bricht den Vergleich ab, statt als neutrale Klassifikation in das F1 einzugehen.
Vorhandene Ausgabedateien verhindern eine versehentliche Wiederholung.

```bash
# Vom Repository-Root; dieser Lauf benötigt eine entsprechende API-Freigabe.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
EMBEDDING_LOCAL_FILES_ONLY=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MOCK_APERTUS=false .venv/bin/python \
  track_2a/scripts/run_hybrid_quality_comparison.py \
  --dataset track_2a/data/evaluation/hybrid_quality_50_validation.jsonl \
  --max-api-requests 200 \
  --output track_2a/docs/evaluation/hybrid_quality_2026-10-05.json
```

Zwei Methoden mit erstem Durchlauf und warmer Wiederholung benötigen regulär
200 API-Anfragen. Bereits vorhandene Indizes werden nicht gelöscht; Cache- und
Indexmetriken zeigen den tatsächlich beobachteten Zustand.
