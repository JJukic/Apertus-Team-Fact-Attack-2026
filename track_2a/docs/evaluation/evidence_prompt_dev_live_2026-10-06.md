# Iteration 4: Live-Vergleich der Beleg-Prompts

Modell: `swiss-ai/Apertus-v1.5-70B-thinking`, durch alle angefragten und zurückgegebenen Modell-IDs bestätigt.
Genau 200 transportseitig erfolgreiche API-Anfragen, kein Budgetübertritt. 50 identische Entwicklungsfälle
für BM25 und hybrid_dense, jeweils einmal mit bisherigem und neuem Prompt.
Retrieval-Kontext, Quellen, Labels und übrige Entscheidungsregeln je Methodenpaar eingefroren.
Kein unabhängiger Test; 32 unterschiedliche Claims und 18 Übersetzungsvarianten.

**Gültigkeitsaudit: Drei Antworten enden mit finish_reason=length.**
Die Scores dokumentieren die alte Verarbeitung, keinen vollständig gültigen Vergleich.
Siehe [Antwortaudit](response_validity_iteration_6.json).

| Methode | Bisheriger Prompt | Neuer Prompt | Falsch → richtig | Richtig → falsch |
|---|---:|---:|---:|---:|
| retrieval | 0.7473 | 0.7705 | 1 | 0 |
| hybrid_dense | 0.7371 | 0.7576 | 4 | 3 |

## Macro-F1 nach Sprache

| Methode | Sprache | Bisherig | Neu |
|---|---|---:|---:|
| retrieval | de | 0.7535 | 0.8202 |
| retrieval | fr | 0.7424 | 0.7424 |
| retrieval | it | 0.7424 | 0.7424 |
| hybrid_dense | de | 0.7222 | 0.7117 |
| hybrid_dense | fr | 0.8121 | 0.8771 |
| hybrid_dense | it | 0.6778 | 0.6848 |

## Konfusionsmatrizen

Zeilen Gold, Spalten Vorhersage, Reihenfolge immer 0 / 1 / 2.

- retrieval / baseline: `[[17, 2, 3], [0, 12, 0], [0, 7, 9]]`
- retrieval / candidate: `[[17, 2, 3], [0, 12, 0], [0, 6, 10]]`
- hybrid_dense / baseline: `[[11, 8, 3], [0, 11, 1], [0, 1, 15]]`
- hybrid_dense / candidate: `[[12, 7, 3], [0, 11, 1], [0, 1, 15]]`

## Belegprüfung

| Methode | Variante | Zitate | Wörtlich auf zugeordneter Textschicht-Seite |
|---|---|---:|---:|
| retrieval | baseline | 36 | 21 |
| retrieval | candidate | 38 | 15 |
| hybrid_dense | baseline | 31 | 31 |
| hybrid_dense | candidate | 37 | 37 |

Die Textschicht-Zuordnung ist kein Nachweis semantischer oder sichtbarer Seitenrichtigkeit.
Die vollständige Audit-Tabelle enthält 234 Zeilen; noch offene Sicht-/Inhaltsprüfungen
stehen ausdrücklich auf `pending`.

## Geänderte Entscheidungen

| Methode | Fall | Vorher | Neu | Gold |
|---|---|---:|---:|---:|
| retrieval | historical-11 | 1 | 2 | 2 |
| hybrid_dense | historical-04 | 1 | 0 | 0 |
| hybrid_dense | historical-06 | 0 | 1 | 0 |
| hybrid_dense | historical-10 | 0 | 1 | 0 |
| hybrid_dense | historical-11 | 1 | 2 | 2 |
| hybrid_dense | historical-12 | 1 | 0 | 0 |
| hybrid_dense | historical-24 | 2 | 1 | 2 |
| hybrid_dense | cross-it-fr-0 | 1 | 0 | 0 |

## Qualitätsentscheidung und Fehlerursachen

**Kein bestätigter Qualitätsgewinn; Prompt nicht produktiv übernommen.**
BM25 gewinnt nur `historical-11`, dessen neues Zitat keine zugeordnete Originalseite hat
und eine umformulierte Tabellenbeschreibung enthält. Seine wörtliche Belegquote verschlechtert sich.
Dense gewinnt vier und verliert drei Entscheidungen. Bei `historical-06` ist das neue Zitat
umgestellt; bei `historical-10` verbindet es Text mit Auslassungen. Beide zuvor richtigen
Entailments werden durch die Belegprüfung zu Neutral. `historical-24` verliert bereits im
Modelloutput: Neutral trotz widersprechender Begründung. Keine Lockerung des Matchers als Reparatur.
Die verbesserten Zitate für `historical-04`, `historical-12` und `cross-it-fr-0`
sind sachlich passend. `historical-20` nutzt diesmal die passende Seite 10.
Bei `historical-11` genügt die allein zitierte Mindest-/Höchstanteilsangabe nicht als
formaler Ausschluss eines 100-Prozent-Kantonsanteils; die gemeinsame Finanzierung muss
im Beleg selbst erkennbar sein.
Zusätzlich steht der bei `cross-fr-de-2` als Seite 33 ausgegebene Mietrecht-Absatz
sichtbar auf Seite 32. Die Textschicht dupliziert ihn auf Seite 33. Der Ausschnitt enthält
zudem nur das schriftliche Gesuch, nicht die Zustimmung. Sichtprüfung von Seiten 32/33 dokumentiert.
Die drei fraglichen übernommenen Gold-Labels wurden nicht geändert; die ergänzende
JSON-Zusammenfassung berechnet die Sensitivität ohne sie separat.

## Nächster Versuch

Zuerst die Seitenzuordnung der PDF-Textschicht separat untersuchen und konservativ prüfen.
Danach ein isoliertes Verfahren zur Auswahl vollständiger, relevanter Originalspans erproben.
Keine neue Live-Schleife und kein Testset-Tuning: die zusätzlich freigegebenen 200 Anfragen
sind ausgeschöpft. Ein belastbarer Gewinn erfordert weiterhin ungenutzte Testdaten.

Reproduktion ohne API aus `track_2a`:

```sh
../.venv/bin/python scripts/summarize_prompt_comparison.py
```
