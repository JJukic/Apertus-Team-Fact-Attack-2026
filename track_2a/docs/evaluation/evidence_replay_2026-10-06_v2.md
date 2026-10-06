# Offline-Nachmessung der Belegprüfung

Gespeicherte Antworten des Live-Vergleichs vom 5. Oktober 2026; keine neuen API-Anfragen.
Je 50 identische Fälle pro Methode und Durchlauf. Neu berechnet werden Zitatzuordnung und erkannte Zahlenkonflikte direkt im zugeordneten Originalzitat.

| Methode | Durchlauf | Macro-F1 vorher | Macro-F1 Replay | Falsch → richtig | Richtig → falsch |
|---|---|---:|---:|---:|---:|
| hybrid | first_pass | 0.6016 | 0.6410 | 2 | 0 |
| hybrid | warm_repeat | 0.5593 | 0.6412 | 4 | 0 |
| hybrid_dense | first_pass | 0.5865 | 0.7194 | 6 | 0 |
| hybrid_dense | warm_repeat | 0.6106 | 0.7396 | 6 | 0 |

## Macro-F1 nach Sprache

| Methode | Durchlauf | DE | FR | IT |
|---|---|---:|---:|---:|
| hybrid | first_pass | 0.546 | 0.7515 | 0.6242 |
| hybrid | warm_repeat | 0.6092 | 0.7515 | 0.5572 |
| hybrid_dense | first_pass | 0.7723 | 0.7455 | 0.6222 |
| hybrid_dense | warm_repeat | 0.7723 | 0.8121 | 0.6222 |

## Konfusionsmatrizen nach Replay

Zeilen = Gold-Label, Spalten = vorhergesagtes Label; Reihenfolge 0, 1, 2.

**hybrid / first_pass**

| Gold \ Vorhersage | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 11 | 7 | 4 |
| 1 | 0 | 11 | 1 |
| 2 | 0 | 6 | 10 |

**hybrid / warm_repeat**

| Gold \ Vorhersage | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 10 | 8 | 4 |
| 1 | 0 | 11 | 1 |
| 2 | 0 | 5 | 11 |

**hybrid_dense / first_pass**

| Gold \ Vorhersage | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 11 | 8 | 3 |
| 1 | 0 | 11 | 1 |
| 2 | 1 | 1 | 14 |

**hybrid_dense / warm_repeat**

| Gold \ Vorhersage | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 12 | 7 | 3 |
| 1 | 0 | 11 | 1 |
| 2 | 1 | 1 | 14 |

## Originalzitat-Zuordnung

| Methode | Durchlauf | Modellzitate | Zugeordnete Originalspans | Fälle mit mindestens einem Span |
|---|---|---:|---:|---:|
| hybrid | first_pass | 54 | 23 | 19/50 |
| hybrid | warm_repeat | 54 | 22 | 18/50 |
| hybrid_dense | first_pass | 47 | 21 | 20/50 |
| hybrid_dense | warm_repeat | 46 | 21 | 20/50 |

## Grenzen

Dies ist eine kontrollierte Nachberechnung, kein neuer Live-Benchmark und kein unabhängiger Test.
Die 14 manuell untersuchten Fälle wurden zur Entwicklung verwendet und sind Teil dieser 50 Fälle.
Die drei fraglichen Gold-Labels bleiben unverändert; die JSON-Datei enthält zusätzlich Werte ohne diese Fälle.
Bei historical-20 wird ein echtes, aber sachfremdes Zitat von Seite 48 akzeptiert; dies verletzt die Qualitätsbedingung.
Die passende Aussage steht auf der ausgewählten Seite 10; das Modell paraphrasiert sie,
weshalb sie nicht wörtlich zugeordnet wird. F1 ist daher nur ein technischer Replay-Befund, kein bestätigter
Qualitätsgewinn. Exakte Zitatzuordnung ist automatisch geprüft; semantische Beleggültigkeit nicht für
den gesamten Testsatz.
Die zwei Durchläufe sind wiederholte Modellantworten auf dieselben Fälle, keine 100 unabhängigen Fälle.
BM25/full sind in diesen gespeicherten Vergleichsjournalen nicht enthalten.

## Geänderte Labels

| Methode | Durchlauf | Fall | Vorher | Replay | Gold |
|---|---|---|---:|---:|---:|
| hybrid | first_pass | historical-07 | 1 | 0 | 0 |
| hybrid | first_pass | historical-29 | 1 | 0 | 0 |
| hybrid | warm_repeat | historical-07 | 1 | 0 | 0 |
| hybrid | warm_repeat | historical-11 | 1 | 2 | 2 |
| hybrid | warm_repeat | historical-16 | 1 | 0 | 0 |
| hybrid | warm_repeat | cross-it-fr-0 | 1 | 0 | 0 |
| hybrid_dense | first_pass | historical-07 | 1 | 0 | 0 |
| hybrid_dense | first_pass | historical-20 | 1 | 0 | 0 |
| hybrid_dense | first_pass | historical-26 | 1 | 0 | 0 |
| hybrid_dense | first_pass | historical-29 | 1 | 0 | 0 |
| hybrid_dense | first_pass | cross-de-it-0 | 1 | 0 | 0 |
| hybrid_dense | first_pass | cross-it-fr-0 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | historical-07 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | historical-20 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | historical-26 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | historical-29 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | cross-de-it-0 | 1 | 0 | 0 |
| hybrid_dense | warm_repeat | cross-it-fr-0 | 1 | 0 | 0 |

Reproduktion aus `track_2a`:

```sh
../.venv/bin/python scripts/replay_evidence_comparison.py --version v2
```

Zusätzlicher Prüfschritt: erkannte Zahlenkonflikte innerhalb eines tatsächlich
zugeordneten Zitats ändern Entailment zu Contradiction. Dieser Schritt fand im Replay
nur bei `historical-11` einen Konflikt. Er prüft keine nichtnumerischen Widersprüche.

