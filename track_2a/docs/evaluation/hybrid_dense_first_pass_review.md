# Manuelle Prüfung der 14 Belegregel-Fehler

Die [Prüftabelle](hybrid_dense_first_pass_manual_review.csv) enthält alle 14 Fälle
aus `hybrid_dense` / `first_pass`, deren Rohlabel mit dem geerbten Gold-Label
übereinstimmte und deren finales Label durch die Belegregel falsch wurde.
Ich habe die gespeicherten Modellzitate mit den **Original-PDFs** vom
24. November 2024 abgeglichen. Die PDFs stimmen per SHA256 mit den im
[Live-Bericht](hybrid_quality_2026-10-05.json) gespeicherten Dateien überein.
Relevante Seiten wurden zusätzlich als PDF-Seitenbilder kontrolliert.

**In allen 14 Fällen enthält mindestens eine der zehn ausgewählten Seiten eine
Textstelle, die den Entailment-Claim stützt.** Keiner dieser 14 Fälle belegt
somit einen vollständigen Retrieval-Ausfall. Diese Aussage gilt nur für diese
14 vorselektierten Belegregel-Fehler, nicht für alle Fälle oder Methoden.

| Ursache des nicht zuordenbaren Modellbelegs | Fälle | Fall-IDs |
|---|---:|---|
| PDF-/Zitatformat (Zeilentrennung, Gedankenstrich) | 5 | `cross-de-it-0`, `cross-it-fr-0`, `historical-07`, `historical-26`, `historical-29` |
| Umformuliertes, übersetztes oder zusammengesetztes Modellzitat | 8 | `cross-fr-it-0`, `cross-it-de-0`, `cross-fr-de-0`, `historical-00`, `historical-06`, `historical-10`, `historical-20`, `historical-30` |
| Kein Modellzitat ausgegeben | 1 | `historical-12` |

Besonders aufschlussreich sind vier Fälle:

- `cross-fr-de-0`: Die hinterlegte Referenzseite 4 fehlt unter den ausgewählten
  Seiten, aber **Seite 22** enthält im Gesetzestext ebenfalls die sechs
  Autobahnabschnitte. `reference_page_selected=false` war hier kein Beweis für
  einen Retrieval-Fehler. Das Modell setzte nicht zusammenhängende Teile des
  Gesetzestextes zu einem vermeintlichen Zitat zusammen.
- `historical-06`: Die Empfehlung zur Annahme der Untermietvorlage steht auf
  **Seite 31**. Das Modell setzte `[Page 30]` davor und verwendete Auslassungen.
- `cross-it-de-0`: Die sechs Abschnitte stehen auf der **deutschen Seite 4**;
  das Modell lieferte dazu einen italienischen Text statt eines Zitats aus dem
  deutschen Büchlein.
- `historical-12`: **Französische Seite 4** belegt den Claim, doch die
  Modellantwort enthält überhaupt kein `evidence`-Element.

Bei `historical-07` und `historical-29` steht die maßgebliche Gesetzesstelle
visuell auf **Seite 42**. Die Textextraktion enthält Teile davon auch für
Seite 43, obwohl sie dort im gerenderten PDF nicht sichtbar sind. Für
zuverlässige Seitenangaben sollte diese PDF-Textlagen-Anomalie separat geprüft
werden. Die hier verifizierte Seite ist deshalb 42.

**Nächste technische Prüfung:** Bei den fünf Formatfällen kann man eine
kontrollierte Normalisierung von PDF-Trennungen und Unicode-Gedankenstrichen
testen, ohne den Quelltext oder die Seiten-ID zu verlieren. Die acht
umformulierten/zusammengesetzten Zitate dürfen dadurch nicht einfach als
Originalzitate akzeptiert werden; besser wäre eine Ausgabe mit Absatz-ID, aus
der die Anwendung einen tatsächlichen Originalspan entnimmt. Der Fall ohne
Modellzitat erfordert eine eigene Prompt-/Antwortprüfung. Erst nach einer
getrennten Validierung auf nicht für die Entwicklung verwendeten Fällen sollte
ein neuer Pipeline-F1 gemessen werden.

Die drei bereits gesondert markierten fraglichen Gold-Labels sind **nicht**
Teil dieser 14 Fälle. Weder Datensatz noch Labels, Pipeline oder Benchmarkscore
wurden für diese Prüfung verändert; es gab keine weiteren API-Anfragen.

## Technische Nachprüfung vom 6. Oktober 2026

Die Belegzuordnung für `hybrid` und `hybrid_dense` verwendet jetzt
`src/evidence_matcher.py`. Beim Vergleich werden Leerraum, Groß-/Kleinschreibung,
weiche Trennzeichen, PDF-Worttrennungen zwischen Buchstaben sowie En-/Em-Dashes
normalisiert. Ein Leerzeichen vor dem Trennstrich wird nur bei anschließendem
Zeilenumbruch entfernt. Zahlenbereiche und mathematische Minuszeichen werden
nicht vereinheitlicht. Die Ausgabe bleibt ein zusammenhängender Ausschnitt
aus dem originalen extrahierten Seitentext; die Seite stammt aus der Quelle.
Eine Modell-Seitenangabe entscheidet nur zwischen bereits verifizierten Treffern.
Es gibt keine Ähnlichkeitsschwelle, Übersetzung oder Zusammenfügung von Seiten.

Der Offline-Test verwendet die gespeicherten Zitate, die damals ausgewählten
Seiten und die per SHA256 geprüften Original-PDFs. Alle fünf Formatfälle sind
jetzt zuordenbar. Der Fall ohne Zitat bleibt ohne Treffer. Die umformulierten
und zusammengesetzten Zitate werden weiterhin abgelehnt.

**Ergänzung zur fallweisen manuellen Einordnung:** `historical-20` enthält zwei
Zitate. Das erste lässt sich nach Normalisierung wörtlich auf der ausgewählten
Seite 48 finden (Koordination der Versorgung). Das zweite, zur einheitlichen
Finanzierungsaufteilung, bleibt eine nicht akzeptierte Paraphrase. Daher haben
insgesamt **6 von 14 Fällen mindestens einen Texttreffer**, aber nur fünf sind
die zuvor ausgewiesenen reinen Formatfälle. Ein Texttreffer allein prüft nicht,
ob das Zitat die Aussage inhaltlich trägt. Die bestehende Pipeline akzeptiert
für die Belegregel mindestens einen Treffer; diese semantische Einschränkung
bleibt bestehen und darf nicht als nachgewiesene Qualitätssteigerung ausgegeben
werden. Die ursprüngliche manuelle Prüftabelle bleibt als Audit erhalten.

Validierung (aus `track_2a`, mit der Projektumgebung):

```sh
../.venv/bin/python -m pytest tests/test_evidence_matcher.py tests/test_hybrid_retriever.py tests/test_pipeline.py -q
```

Ergebnis: **47 Tests und 27 Untertests bestanden**, einschließlich aller 14
historischen Fälle, falscher Zahlen, ausgelassener Negation, Wortgrenzen,
fehlender Belege, falscher Seiten-Tags und Integration beider Hybrid-Modi.
Fehlen historische PDFs, wird der PDF-Regressionstest ausdrücklich übersprungen;
die unabhängigen Matcher-Tests laufen trotzdem.

Keine neuen API-Anfragen, kein neuer Live-F1. Retrieval-Ranking und Prompts sind
unverändert. BM25 (`retrieval`), `full` und `direct_reference` verwenden weiterhin
ihre bisherige Beleglogik; der nächste Methodenvergleich muss diese Unterschiede
berücksichtigen. Eine semantische Prüfung der Aussage gegen den Beleg und die
PDF-Textlagen-Anomalie bleiben separate Aufgaben.

## Offline-Nachmessung

Der [Replay-Bericht](evidence_replay_2026-10-06_v2.md) vergleicht alle 200
gespeicherten Antworten (50 Fälle × 2 Methoden × 2 Durchläufe) mit der neuen
Belegprüfung. Die ursprünglichen Live-Berichte bleiben unverändert.
Die Labels und zugeordneten Originalzitate des Replays wurden zusätzlich für
alle 200 Antworten gegen die tatsächliche Belegstufe in `verify_claim` getestet.
Dabei waren alle vorgelagerten Entscheidungen eingefroren und API-Aufrufe
durch gespeicherte Antworten ersetzt.

```sh
../.venv/bin/python -m pytest tests/test_evidence_replay.py tests/test_evidence_matcher.py -q
```

Die gezielte Replay-Prüfung stimmt für alle 200 Antworten mit `verify_claim`
überein. Die vollständige Suite umfasst 76 Tests und 227 Untertests; sie prüft
die Pipeline, nicht die semantische Belegqualität aller 50 Fälle.

## Iteration 2: Zahlenkonflikt direkt im belegten Zitat

Die erste Replay-Version änderte `historical-11` in `Entailment`, obwohl das
Modellzitat auf Seite 10 die Behauptung ausdrücklich widerlegt: Der Kanton
übernimmt mindestens 55 Prozent und die Krankenkasse den Rest. Der vorherige
Zahlenprüfer sah den gesamten Kontext. Dort konnte eine gleiche Zahl zu einem
anderen Thema den Konflikt verdecken.

Für `hybrid` und `hybrid_dense` prüft die Pipeline deshalb erkannte
Zahlenkonflikte nach der wörtlichen Zuordnung erneut direkt im zitierten
Originalspan. Im Offline-Replay wurde nur `historical-11` dadurch von Label 0
auf Label 2 korrigiert. Hybrid / warm-repeat steigt gegenüber dem ersten
Replay von F1 0,6208 auf 0,6412; die anderen drei Durchlaufwerte bleiben
gleich. Im selben Fall wechselt der Fehler von falsch vorhergesagt Neutral zu
richtig vorhergesagt Contradiction.

Die manuelle Prüfung der acht unterschiedlichen Fälle mit Labeländerungen
steht in der [Fallprüfung](evidence_replay_2026-10-06_v2_manual_audit.csv).
Sie bestätigt sieben relevante Zitate; bei `historical-20` ordnet der Matcher
jedoch ein wörtliches, aber sachfremdes Zitat von Seite 48 zu. Das Büchlein
stützt die Behauptung auf Seite 10, doch das Modell hat diese Passage nur
paraphrasiert. Dieser Fall verletzt die Bedingung, dass die akzeptierte
Belegstelle die Aussage stützen muss. Der Replay-F1 ist somit ein technischer
Befund, keine bestandene Qualitätsverbesserung.

Der Zahlencheck ist eine gezielte Korrektur und besteht die Regressionstests.
Die sachfremde Zitatzuordnung muss noch behoben werden. Danach braucht es eine
neue Evaluation auf einem separaten, nicht zur Entwicklung verwendeten
Datensatz. Bis dahin sind die 50 Fälle nur Entwicklungsdaten.
