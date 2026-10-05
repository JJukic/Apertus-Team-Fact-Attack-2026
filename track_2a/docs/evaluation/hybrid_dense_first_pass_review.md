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
