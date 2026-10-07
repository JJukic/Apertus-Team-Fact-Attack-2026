# Adaptive Multilingual Hybrid Retrieval with NLI-Aware Reranking

Status: Umsetzung und Validierungsarbeit laufen auf
`feat/adaptive-multilingual-nli-reranking`, Ausgangscommit `f10bbad`.
Diese Methode ist noch nicht als besser nachgewiesen und ersetzt keinen Default.

Die neue Zielvorgabe fordert ausdrücklich neue kontrollierte Ablationen.
Die zuvor gestoppten Experimente bleiben auf `feat/hybrid-dense-retrieval`
archiviert und werden nicht weitergeführt.

## Qualitätsmaßstab

Primär entscheidet Macro-F1 über Entailment, Neutral und Contradiction.
BM25 bleibt die zu schlagende Baseline. Weniger Tokens oder geringere Latenz
rechtfertigen keinen schlechteren F1. Modellkomplexität ist kein Auswahlkriterium.
Die finale Auswahl erfolgt nur auf Entwicklungs-/Validierungsdaten; danach wird
eine eingefrorene Konfiguration einmal auf bisher ungenutzten Büchlein-Daten geprüft.

## Daten und Belege

`EvidenceUnitParser` extrahiert Layout-Blöcke statt gleich großer Textfenster.
Einheiten besitzen Dokumenthash, physische 1-basierte Seite, stabile Paragraph-ID,
Abschnitt, Sprache, Bounding Box, Originaltext und Lesereihenfolge.
Originalzeilen, Listen und Zahlen bleiben erhalten. Nichtmalende, transparente
und außerhalb der Seite liegende Spannen werden ausgeschlossen; getrennte
Spannen werden niemals über solche Lücken zu einem Zitat verbunden.
Beliebige Überdeckung und gescannte Tabellen benötigen zusätzliche Sichtprüfung.

Die PyMuPDF-Span-Informationen und ihre Sichtbarkeitsmerkmale sind in der
[offiziellen Dokumentation](https://pymupdf.readthedocs.io/en/latest/textpage.html)
beschrieben. Der Parser und seine Bibliotheksversion werden in Ergebnissen erfasst.

Die offiziellen OST-Referenzstrings enthalten häufig ganze Abschnitte statt
einzelner entscheidender Sätze. Deshalb sind zwei Messgrößen zu unterscheiden:

- **Referenzbereich-Recall:** eine Einheit ist dem offiziellen Referenzabschnitt
  textuell zugeordnet. Ein Treffer kann noch inhaltlich unzureichend sein.
- **Gold Evidence Recall:** Treffer auf überprüfte entscheidende Einheiten oder
  erforderliche Gruppen mehrerer Einheiten. Nur vorhandene überprüfte Annotationen
  zählen hierzu; fehlende Annotationen werden als fehlend ausgewiesen.

Goldlabels, Goldreferenzen und annotierte Gold-IDs dürfen nur die Auswertung
und Annotation beeinflussen, niemals die Advanced-Vorhersage.

## Module

1. Sprache des Claims und des Dokuments feststellen; bereitgestellte Sprache
   bleibt eine explizite Aufgabeneingabe.
2. Originalclaim und treue DE/FR/IT-Suchübersetzungen vorbereiten und cachen.
3. Original-BM25, übersetztes BM25 und mehrsprachige Dense-Suche parallel und
   unabhängig ausführen, zunächst je Top 20.
4. Vollständige Union über Original-IDs deduplizieren. Keine Score-Fusion und
   kein Kandidatenverlust durch eine zusätzliche Union-Grenze.
5. Optionales NLI-Reranking; bestätigende und widersprechende Passagen zählen
   als relevant. Top-K Einheiten auswählen.
6. Konfigurierbar ±0, ±1 oder ±2 benachbarte Textparagraphen ergänzen; Überschriften
   verbrauchen den Radius nicht. Abschnitts- und Seitenangaben bleiben an jeder
   Einheit erhalten. Eine Variante begrenzt die Erweiterung auf denselben Abschnitt.
7. Apertus klassifiziert ausschließlich anhand dieses Quellenkontexts.
   Belegtexte kommen über Original-IDs aus dem Code, nie aus generierten Zitaten.
8. Optionale Neutral-Kalibrierung mit auf Validierungsdaten gesuchten Schwellen.

Für den lokalen Reranker ist zunächst
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` vorgesehen,
unveränderliche Revision `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`.
Seine Trainingssprachen enthalten DE, FR und IT; das ist eine Eignungshypothese,
kein Nachweis der Qualität auf Abstimmungsbüchlein.
Siehe die [Modellkarte des Autors](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).
Die Passage ist die Prämisse und der Claim die Hypothese. Der Relevanzscore ist
`p(entailment) + p(contradiction)`. Lange Einheiten können mehrere lokale
Modelleingabefenster verwenden; die Originaleinheit wird dadurch nicht zerschnitten.

MMR ist ausgeschaltet. RRF ist nur eine explizite Vergleichsvariante.
Alle API-Aufrufe werden gezählt, einschließlich Übersetzung, binärer NLI-Pässe,
Konfliktentscheidung und Fehlversuche. Kein Output-Tokenlimit wird gesetzt.
Technische Fehler erhalten kein Neutral-Label.

## Verbindliche Ablationen

| ID | Konfiguration |
|---|---|
| E0 | Gesamtdokument → Apertus |
| E1 | Original-BM25 → Apertus |
| E2 | Original-BM25 + Nachbarn → Apertus |
| E3 | Original-BM25 + übersetztes BM25 → Apertus |
| E4 | Original-BM25 + übersetztes BM25 + Dense-Union → Apertus |
| E5 | Union + NLI-Reranker → Apertus |
| E6 | Union + NLI-Reranker + Nachbarn → Apertus |
| E7 | E6 mit validierter Neutral-Schwelle |
| E8 | Direkte Dreiklassen-NLI gegenüber Support-/Contradiction-Pässen |

Zusätzliche Retrieval-Vergleiche: reine Dense-Rangfolge, RRF, BM25-Parameter und
Nachbarschaftsradien 0/1/2. Komponenten bleiben nur bei belegtem Qualitätsnutzen.

## Abschlussnachweise

| Anforderung | Erforderlicher Nachweis | Stand |
|---|---|---|
| CLI-kompatible Gesamtpipeline | echter Aufruf, maschinenlesbares Label und Originalbelege | offen |
| Paragraph-/Seitenprovenienz | Quellen-IDs, Textidentität und sichtbare Stichproben | Parser implementiert, Prüfung läuft |
| drei unabhängige Generatoren | getrennte Ranglisten und Recall@1/3/5/10/20 | implementiert, Messung offen |
| Union mit hohem Gold-Recall | überprüfte Goldannotation, Recall und Abdeckung | offen |
| NLI-Reranker | Contra-/Entailment-Test, Goldrank, MRR, Ablation | implementiert, Messung offen |
| Nachbarn | Radius 0/1/2, unveränderte IDs und Grenzen | implementiert, Prüfung offen |
| Apertus-NLI | strikte JSON-Ausgabe, Quellenbindung, keine Fehlerlabels | offen |
| Neutral-Kalibrierung | gespeicherte Schwellen-Suche auf Validation | offen |
| volle Ablationen | E0–E8, gleiche Fälle, F1 je Klasse, Konfusionsmatrix | offen |
| Fehleranalyse | Kategorien und tatsächliche Fehlerbeispiele | offen |
| Sprachvergleich | DE/FR/IT, neun Paare, same-/cross-language | offen |
| finale Auswahl/Test | Validation-Auswahl, Freeze, einmaliger Test | offen |

Noch keine neue Macro-F1-Aussage. Ein guter Architekturentwurf oder bestandene
Unit-Tests allein erfüllen diese Abschlussanforderungen nicht.
