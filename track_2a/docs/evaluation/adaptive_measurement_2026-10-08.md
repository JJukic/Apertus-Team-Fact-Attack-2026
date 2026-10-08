# Adaptive Macro-F1-Messung vom 8. Oktober 2026

Die Messung wurde auf ausdrücklichen Wunsch des Benutzers wieder aufgenommen.
Der archivierte Stand vom 7. Oktober bleibt erhalten. Aktueller lokaler Lauf:
`adaptive/run_2026-10-08_cscs_json_v3`. Weitere Ablationen laufen noch;
eine finale Konfiguration ist noch nicht ausgewählt.

## Daten und Modell

- Validation: 108 Fälle zum Büchlein vom 24. November 2024.
- Goldlabels: 34 Entailment, 42 Neutral, 32 Contradiction.
- 42 same-language und 66 cross-language Fälle, neun Sprachpaare.
- Apertus: `swiss-ai/Apertus-v1.5-70B-thinking`, Temperatur 0,
  kein gesetztes Output-Tokenlimit, vier parallele lokale Clients.
- Die 202 separaten Testfälle bleiben bis zur eingefrorenen Auswahl ungenutzt.

## Vollständig gemessene Varianten

Alle folgenden Scores verwenden dieselben 108 Fälle mit 100 % Abdeckung.

| Variante | Macro-F1 | Same-language | Cross-language |
|---|---:|---:|---:|
| E1: Original-BM25 | 0.4666 | 0.6102 | 0.2594 |
| E6: Union + NLI-Reranking + Nachbarn | 0.7397 | 0.7049 | 0.7456 |
| E7: E6 + Neutral-Kalibrierung | 0.8517 | 0.7517 | 0.9005 |

E7 wählt auf Validation `best_reranker_score < 0.7535629272460938`.
Dies korrigiert gegenüber E6 14 Neutral-Fälle; ein korrektes Entailment
wird dabei Neutral. **Der Kalibrierungswert ist kein unabhängiger Testscore.**

Die frühere perfekte Juni-Messung verwendet andere Fälle und eine andere
Pipeline. Sie ist keine kontrollierte Baseline für diese November-Messung
mit Layout-Belegeinheiten, Top 5 und dem gemeinsamen neuen NLI-Protokoll.

## Lokale Ergebnisse und Wiederaufnahme

Die umfangreichen Artefakte sind per `.gitignore` ausgeschlossen und bleiben
lokal. Git enthält nur Code, Tests und kompakte Berichte.

- [Live-Fortschritt](adaptive/run_2026-10-08_cscs_json_v3/validation/progress.md)
- [Ergebnistabelle](adaptive/run_2026-10-08_cscs_json_v3/validation/results.md)
- `validation/results.json`: Konfusionsmatrizen, F1 je Klasse und Sprachpaare.
- `validation/predictions/` und `*.cases.jsonl`: einzelne Vorhersagen.
- `validation/requests.api.jsonl`: alle API-Versuche und Antworten.
- `validation/run.json`: Quellensnapshots, Datensatzhashes, Modelle und Prompts.

Vom Repository-Verzeichnis aus, bei unveränderten Quellen:

```bash
export EMBEDDING_BATCH_SIZE=4
export EMBEDDING_CACHE_DIR=/iopsstor/scratch/cscs/jjukic/apertus-f1-2026-10-08/repo/track_2a/.cache/embeddings
.venv/bin/python track_2a/scripts/run_adaptive_comparison.py summarize \
  --output track_2a/docs/evaluation/adaptive/run_2026-10-08_cscs_json_v3 --device cuda
```

Die CUDA-Angabe und der Cachepfad bewahren die vorbereitete Konfiguration;
`summarize` lädt keine GPU-Modelle. `run` verwendet vorhandene Vorbereitungen
und erfolgreiche Vorhersagen erneut. Während des laufenden Prozesses darf
keine zweite Inferenz für denselben Ergebnisordner gestartet werden.

## CSCS und Belegprüfung

Die SSH-Anmeldung funktioniert gemäß der [CSCS-Anleitung](https://docs.cscs.ch/access/ssh/).
Die GPU-Vorbereitung lief auf Clariden, Account `g234`, QoS `large-sc-qos`,
Partition `normal`, PyTorch-uenv `c05f143d5fbf0927`.
Aktueller Job **3614229**: `COMPLETED`, Exitcode `0:0`, Dauer **1:24 Minuten**,
GPU NVIDIA GH200 120GB, Torch 2.9.1, CUDA 12.9.
Arbeitsverzeichnis: `/iopsstor/scratch/cscs/jjukic/apertus-f1-2026-10-08`.

Nur benötigte Quellen, Validation-Daten, drei PDFs und Suchübersetzungen
wurden übertragen. API-Key und `.env` blieben lokal. Beide Suchmodelle
werden mit festgelegten Revisionen offline verwendet. Der Import hat alle
108 Vorbereitungen sowie Originaltexte, Seiten und IDs der drei lokalen
PDFs verifiziert. GPU- und Client-Versionen stehen in `gpu_runtime.json`
und `client_runtime.json` im lokalen Ergebnisordner.

51 eindeutige sprachübergreifende Suchübersetzungen wurden vorbereitet;
das lokale Journal liegt in `adaptive/queries_2026-10-08.api.jsonl`.
Die Belegauswertung benötigt keine neuen API-Aufrufe. Bei **drei** überprüften
Goldfällen enthält E6 nach Nachbarn alle benötigten Beleggruppen (3/3),
E1 in 2/3 Fällen. Diese kleine Stichprobe beweist keine allgemeine
Retrievalqualität. Details stehen in `validation/retrieval_results.json`.

## Ausgabevertrag und Grenzen

Der Endpoint akzeptiert `response_format={"type":"json_object"}`.
Der Client berücksichtigt nur vollständiges JSON nach einem gegebenenfalls
vollständigen Thinking-Umschlag. Unbekannte Beleg-IDs, fehlende Pflichtfelder,
unvollständige Antworten und inkonsistente Beleglisten werden abgelehnt.
Bei einem ungültigen Ausgabevertrag wird einmal auf demselben Originalkontext
eine neue Modellantwort angefordert; beide Anfragen werden protokolliert.
Der Code repariert keine Labels. Bleibt die Ausgabe ungültig, bleibt der
Fall ein technischer Fehler. Die Regel gilt für alle Varianten.

Die 31 betroffenen Tests und neun Unterprüfungen bestehen, einschließlich
Wiederaufnahme und Ablehnung veränderter Belegtexte. Ein unvollständiger
Score darf keine Konfiguration auswählen. Die gemeinsam genutzte Vorbereitung
in den Laufzeiten ist kein isolierter Komponentenvergleich; API-Gesamtkosten
müssen alle Versuche aus dem Journal berücksichtigen.

Die [erste Fehlerprüfung](adaptive_error_review_2026-10-08.md) beschreibt
Neutral-Verwechslungen, vertauschte Zahlenzuordnungen und einen durch die
Kalibrierung verlorenen korrekten Fall. Vorhersagen und Goldlabels wurden
für diese Prüfung nicht verändert.

## Bereinigung

Vier abgebrochene Vorversuche wurden auf Wunsch des Benutzers entfernt:
**485 Dateien, 28,5 MiB**. Die Bilanz bleibt hier erhalten.

| Entfernter Ordner | Dateien | Vorhersagen | Gültig | API-Versuche | Eingabe-/Ausgabetokens |
|---|---:|---:|---:|---:|---:|
| `run_2026-10-08` | 11 | 0 | 0 | 0 | 0/0 |
| `run_2026-10-08_cpu` | 27 | 0 | 0 | 0 | 0/0 |
| `run_2026-10-08_cscs` | 156 | 13 | 8 | 13 | 18951/5357 |
| `run_2026-10-08_cscs_json_v2` | 291 | 160 | 133 | 160 | 209912/26860 |

Die MPS-/CPU-Versuche wurden wegen lokalem Speicherdruck beendet.
Die ersten API-Versuche fanden Thinking-/JSON-Formatprobleme und fehlende
Beleglisten. Ihre Rohdaten werden für den korrigierten v3-Lauf nicht benötigt;
die Fehler werden durch Quellcode und Tests nachvollziehbar abgedeckt.
