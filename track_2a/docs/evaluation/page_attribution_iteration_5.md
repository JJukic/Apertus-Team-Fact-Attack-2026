# Iteration 5: Seitenzuordnung außerhalb des sichtbaren PDF-Bereichs

**Offline-Diagnose abgeschlossen; keine neue Modellantwort, keine Labeländerung,
kein nachgewiesener F1-Gewinn.**

## Ursache und einzelne Änderung

Die pypdf-Textlage der deutschen und französischen Büchlein enthält auf Seite 33
zusätzlich Text, der sichtbar auf Seite 32 steht. Koordinaten einzelner Fragmente
liegen außerhalb des Seitenbereichs, beispielsweise x = -312 auf einer Seite
mit x von 0 bis 419,528. Auch PDFKit liest diesen Text; ein bloßer Wechsel zu
dieser Bibliothek behebt die Zuordnung nicht.

Der experimentelle Filter in `scripts/check_page_attribution.py` berücksichtigt
die kombinierte Text-/Benutzermatrix und verwirft Fragmente, deren Ursprung
außerhalb der CropBox liegt. Abstände bleiben erhalten; verworfene Fragmente
werden durch eine explizite Trennung ersetzt, damit keine neuen Zitate durch
Zusammenkleben entstehen. Der erste Versuch verlor gültige Zitate wegen
fehlender Leerraumfragmente; diese Rekonstruktion wurde korrigiert.

Der Filter bleibt außerhalb des produktiven Parsers. Er behandelt noch keine
beliebigen Clip-Pfade, unsichtbaren Zeichenmodi oder partiell sichtbaren Zeilen.
Die [pypdf-Dokumentation](https://pypdf.readthedocs.io/en/6.12.0/user/extract-text.html)
weist zudem auf mögliche Koordinatenfehler bei komplexen Dokumenten hin.

## Reproduzierbares Ergebnis

Ausgangspunkt: eingefrorene Antworten des 200-Anfragen-Promptvergleichs, dessen
50 Fälle bereits Entwicklungsdaten sind. Nur Belegzuordnung wird untersucht.
Alternativseiten sind auf die ursprünglich ausgewählten Seiten beschränkt.

| Methode | Variante | Fall | Bisherige Seite | Gefilterte Seite |
|---|---|---|---:|---:|
| retrieval | baseline | historical-17 | 33 | 32 |
| retrieval | candidate | historical-17 | 33 | 32 |
| hybrid_dense | baseline | historical-05 | 33 | 32 |
| hybrid_dense | candidate | historical-05 | 33 | 32 |
| hybrid_dense | candidate | historical-26 | 33 | 32 |
| hybrid_dense | baseline | cross-fr-de-2 | 33 | 32 |
| hybrid_dense | candidate | cross-fr-de-2 | 33 | 32 |
| hybrid_dense | baseline | cross-it-fr-2 | 33 | 32 |
| hybrid_dense | candidate | cross-it-fr-2 | 33 | 32 |

Die sichtbaren deutschen und französischen Seiten 32/33 wurden gerendert und
geprüft. Alle übrigen bisher zugeordneten Zitate dieses Vergleichs bleiben
zuordenbar. Die neun Änderungen betreffen fünf Fall-/Methodenkombinationen,
keine neun unabhängigen Claims.

Labels bleiben eingefroren: null Korrekturen, null Verschlechterungen. Macro-F1,
Sprachwerte und Konfusionsmatrizen bleiben entsprechend unverändert; sie sind
in `page_attribution_diagnostic_2026-10-06.json` als eingefrorene Kennzahlen
enthalten. Ein vollständig mit neuem Parser ausgeführter Pipeline-F1 wurde
**nicht** gemessen. Sachliche Beleggültigkeit folgt nicht allein aus der richtigen
Seite; insbesondere ersetzt eine Antragspflicht keine Zustimmungspflicht.

## Tests und nächster Schritt

Zwei Regressionstests bestehen: echte DE-Passage von Seite 32 ist nicht mehr auf
Seite 33 auffindbar; alle 68 gespeicherten Dense-Zitate bleiben zuordenbar. Davon werden sieben
ausschließlich von 33 auf 32 verlegt, 61 behalten ihre Zuordnung.

```sh
# Aus track_2a, ohne Apertus-Aufrufe:
../.venv/bin/python scripts/check_page_attribution.py
../.venv/bin/python -m unittest discover -s tests -p test_page_attribution.py -v
```

Vier zusätzliche Tests mit synthetischen PDFs bestehen: Off-Page-Text,
Transformation und CropBox, keine Verbindung über entfernte Fragmente sowie
nicht malende Textmodi 3/7 mit Wiederherstellung des Grafikzustands.
Insgesamt bestehen sechs Tests. Beliebige Clip-Pfade und Transparenz werden
noch nicht vollständig berücksichtigt; vor einer Integration braucht es
eine breitere Sichtprüfung.
Danach wird nur dieser Parserbaustein separat bewertet. Prompt-, Retrieval- oder
Nachsuchänderungen gehören in spätere, getrennte Experimente. Ein unbenutztes
Testset und weitere, ausdrücklich budgetierte Live-Auswertung stehen weiterhin aus.

Ausgangscommit: `c0bd757e19450963bcb7d433166ba1815cbd59b3`.
Die experimentellen Änderungen sind uncommitted; SHA256-Werte der verwendeten
Dateien stehen im Diagnose-JSON.
