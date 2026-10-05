# Qualitätsvergleich Hybrid Retrieval – 5. Oktober 2026

**Der Live-Vergleich ist abgeschlossen.** Vektorsuche verbessert in dieser Probe
sprachübergreifendes F1 in beiden Durchläufen. Beim gesamten Pipeline-F1 fällt der
erste Lauf etwas schlechter und die warme Wiederholung etwas besser aus.
Ein stabiler Gesamtvorteil ist damit noch nicht nachgewiesen; der bisherige Standard
`retrieval` bleibt unverändert.

## Durchführung und Geltungsbereich

- Modell: `swiss-ai/Apertus-v1.5-8B`, echter CSCS-Endpunkt, Temperatur 0, maximal 512 Output-Tokens.
- Genau **200 erfolgreiche API-Anfragen**, keine API-/Parse-Fehler, keine weiteren
  Benchmark-Aufrufe. Die freigegebene Obergrenze von 200 wurde eingehalten.
- Je Methode dieselben 50 Fälle für einen ersten Lauf und eine warme Wiederholung.
  Es sind 32 historische gleichsprachige Fälle und 18 zusätzliche Sprachpaar-Varianten,
  insgesamt **32 verschiedene Claims**. Die sprachübergreifenden Varianten stammen
  von neun bestehenden Claims; je unterschiedlichem Sprachpaar liegen nur drei Fälle vor.
- Historische Büchlein vom 24. November 2024 in DE/FR/IT. Der offizielle Juni-Testsplit
  wurde nicht verwendet. Keine Konfiguration wurde anhand dieser Scores ausgewählt.
- BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`, CPU, vier Threads, Batch 4, Maximalänge 2048,
  40 Seitenkandidaten je Kanal, zehn finale Originalseiten, RRF-Konstante 60.
  Gewichte lagen lokal vor; Embedding-Netzwerkzugriff war ausgeschaltet.
- Die Ausgangsdaten enthalten keine verlässlichen Vorlagentitel. Beide Methoden
  erhalten `vote=None`; die doppelte Titelgewichtung wird hier nicht untersucht.
- Alle gespeicherten SHA256-Prüfsummen der Inference-, Retrieval- und Evaluator-Dateien
  blieben über den Lauf unverändert. Jede vollständige Phase enthält dieselben Fall-IDs.
- Gesamtdauer inklusive Indexaufbau und Wiederholungen: **19.36 Minuten**.
  Gemeldeter Verbrauch: **1,404,724 Input-Tokens** und
  **40,807 Output-Tokens**. Ein Geldbetrag wurde nicht ermittelt.

Der neue [Validierungsdatensatz](../../data/evaluation/hybrid_quality_50_validation.jsonl)
ist eine explorative Regression-/Qualitätsprobe mit abgeleiteten Fällen, kein
unabhängig annotierter Test mit 50 unterschiedlichen Claims. Herkunft, Auswahl und
Gold-Seiten sind in der [Datensatzbeschreibung](../../data/evaluation/README.md) dokumentiert.
Die älteren veröffentlichten F1-Werte für `retrieval` auf anderen Datensätzen sind
mit diesem Vergleich der Methoden `hybrid` und `hybrid_dense` nicht direkt vergleichbar.

## Pipeline-F1 auf den 50 Fällen

Macro-F1 berücksichtigt alle drei Klassen gleich. „Sprachübergreifend“ bedeutet
Claim-Sprache ungleich tatsächlich verwendeter Büchleinsprache; die Teilmenge enthält 18 Fälle.
Die warme Wiederholung ist ein zweiter Modellaufruf je Fall, keine neue Stichprobe.

| Methode | Durchlauf | Macro-F1, 50 Fälle | Sprachübergreifendes F1, 18 Fälle |
|---|---|---:|---:|
| BM25 (`hybrid`) | Erster Lauf | 0.6016 | 0.5333 |
| BM25 (`hybrid`) | Warme Wiederholung | 0.5593 | 0.4524 |
| BM25 + Vektoren (`hybrid_dense`) | Erster Lauf | 0.5865 | 0.6639 |
| BM25 + Vektoren (`hybrid_dense`) | Warme Wiederholung | 0.6106 | 0.6639 |

Die Differenz `hybrid_dense − hybrid` beträgt im ersten Lauf **−0.0151** und in der
warmen Wiederholung **+0.0513**. Sprachübergreifend beträgt sie **+0.1306** bzw.
**+0.2115**. Zwischen den Wiederholungen wechseln 2
BM25-Klassifikationen und 1 Vektorklassifikation.
Es werden weder die Wiederholungen als unabhängige Fälle gezählt noch die besten
Einzelergebnisse als alleiniger Qualitätsnachweis gewählt.

## Problematische übernommene Referenzlabels

Drei bestehende Claims nennen 5,3 Milliarden Franken und sind als Entailment markiert.
Die betreffenden Büchlein nennen auf Seite 4 jedoch 4,9 Milliarden. Der Live-Lauf
verwendete unverändert die übernommenen Labels. Die
[gesonderte Labelprüfung](hybrid_quality_label_audit.json) nennt Claims und Originaltext.
Dies ist keine vollständige unabhängige Nachannotation der Ausgangsdaten.

Zur Transparenz folgen zusätzliche Berechnungen auf denselben gespeicherten
Vorhersagen, **ohne neue API-Anfragen**: einerseits Ausschluss genau dieser drei
fraglichen Fälle, andererseits ihre explizite Korrektur auf Contradiction. Die
primäre 50-Fälle-Auswertung oben bleibt erhalten. Diese ergänzende Prüfung wurde
nach Beobachtung des Problems erstellt und ist ausdrücklich keine vorab registrierte Auswertung.

| Methode | Durchlauf | Gleichsprachiges F1, 32 Fälle | F1 ohne drei fragliche Labels, 47 Fälle | F1 mit drei korrigierten Labels, 50 Fälle |
|---|---|---:|---:|---:|
| BM25 (`hybrid`) | Erster Lauf | 0.6200 | 0.6454 | 0.6622 |
| BM25 (`hybrid`) | Warme Wiederholung | 0.5891 | 0.6009 | 0.6177 |
| BM25 + Vektoren (`hybrid_dense`) | Erster Lauf | 0.5281 | 0.6274 | 0.6326 |
| BM25 + Vektoren (`hybrid_dense`) | Warme Wiederholung | 0.5606 | 0.6531 | 0.6584 |

## Ergebnisse je tatsächlichem Sprachpaar

Je unterschiedlichem Sprachpaar liegt nur ein Beispiel je Klasse vor. Hohe oder
niedrige Einzelwerte haben deshalb eine geringe statistische Aussagekraft.

| Claim → Büchlein | Fälle | BM25 erster Lauf | BM25 warm | Vektoren erster Lauf | Vektoren warm |
|---|---:|---:|---:|---:|---:|
| de->de | 12 | 0.5000 | 0.5000 | 0.5582 | 0.5582 |
| de->fr | 3 | 0.1667 | 0.1667 | 1.0000 | 1.0000 |
| de->it | 3 | 0.5556 | 0.5556 | 0.5556 | 0.5556 |
| fr->de | 3 | 0.5556 | 0.5556 | 0.5556 | 0.5556 |
| fr->fr | 10 | 0.9153 | 0.8024 | 0.6984 | 0.8024 |
| fr->it | 3 | 0.1667 | 0.1667 | 0.5556 | 0.5556 |
| it->de | 3 | 0.1667 | 0.1667 | 0.5556 | 0.5556 |
| it->fr | 3 | 1.0000 | 0.5556 | 0.5556 | 0.5556 |
| it->it | 10 | 0.4722 | 0.4722 | 0.3016 | 0.3016 |

## Belege und Retrieval

Nur zwölf sprachübergreifende Nicht-Neutral-Fälle haben geprüfte Referenzseiten.
Die Listen benennen je **eine hinreichende Seite**, keine vollständige Menge aller
zulässigen Belegseiten. Ein korrektes Zitat einer anderen Seite kann daher einen
Referenzseitenabgleich verfehlen. Die folgenden Raten messen die Abdeckung genau
dieser Referenzen und ersetzen keine semantische Belegprüfung.

| Methode | Durchlauf | Referenzseite im Retrieval-Kontext | Referenzseite im ausgegebenen Beleg | Lexikalische Alignment-Näherung |
|---|---|---:|---:|---:|
| BM25 (`hybrid`) | Erster Lauf | 0.4167 | 0.0833 | 0.2500 |
| BM25 (`hybrid`) | Warme Wiederholung | 0.4167 | 0.0000 | 0.0833 |
| BM25 + Vektoren (`hybrid_dense`) | Erster Lauf | 0.7500 | 0.1667 | 0.1667 |
| BM25 + Vektoren (`hybrid_dense`) | Warme Wiederholung | 0.7500 | 0.1667 | 0.1667 |

Die Vektormethode liefert **9 von 12** geprüften Referenzseiten im Kontext, BM25
**5 von 12**. Die Ausgabe enthält wesentlich seltener passende Referenzseiten.
Die Wortüberlappungsprüfung bleibt eine Näherung (≥ 30 % oder Teilstring), keine
semantische Qualitätsmessung. Der vorhandene Evaluator rundet seinen Seiten-Recall
auf zwei Nachkommastellen; die obige zusätzliche Berechnung verwendet vier.
Gold-Labels, Referenztext und Gold-Seiten wurden ausschließlich nach der Vorhersage
für die Bewertung genutzt, nie als Retrieval- oder NLI-Eingaben.

## Diagnose der nachgelagerten Regeln

Diese zusätzliche Analyse verwendet die bereits gespeicherten API-Antworten. Sie
ändert die gemessene Pipeline nicht und erzeugt keinen zusätzlichen Modellaufruf.
„Rohlabel“ bedeutet das JSON-Label vor Konfidenzregeln und den Pipeline-Guardrails;
sein F1 ist kein Nachweis einer gültigen Belegzuordnung.

| Methode | Durchlauf | F1 der rohen Apertus-Labels | F1 nach Konfidenzregeln | Tatsächliches Pipeline-F1 | Vom Rohlabel abweichende Endlabels |
|---|---|---:|---:|---:|---:|
| BM25 (`hybrid`) | Erster Lauf | 0.8122 | 0.7888 | 0.6016 | 13 |
| BM25 (`hybrid`) | Warme Wiederholung | 0.8122 | 0.7888 | 0.5593 | 15 |
| BM25 + Vektoren (`hybrid_dense`) | Erster Lauf | 0.9044 | 0.9044 | 0.5865 | 15 |
| BM25 + Vektoren (`hybrid_dense`) | Warme Wiederholung | 0.8832 | 0.8832 | 0.6106 | 13 |

In den Antworten kommen inkonsistente Konfidenzen vor, etwa `label=2` bei
`p_neutral=1`. Die bestehenden Entscheidungsregeln können daraus Neutral machen.
Außerdem liefert Apertus teilweise umformulierte Zitate, Auslassungen, Seitenpräfixe
oder geglättete PDF-Zeilentrennungen. Wenn sich daraus kein exaktes Zitat einer
selektierten Originalseite zuordnen lässt, verwirft die Quellenprüfung den Beleg;
Entailment ohne gültigen Beleg wird Neutral. Die rohen Labels sind daher häufig
besser als das tatsächliche, mit Quellen geprüfte Gesamtergebnis.

Weitere Entwicklung sollte die Referenzlabels prüfen, konsistente Originalzitate
und eine verlässliche Zuordnung trotz PDF-Layout untersuchen sowie den Umgang mit
inkonsistenten Konfidenzen evaluieren. Diese Änderungen wurden **nicht während des
Vergleichs** eingebaut. Ein größerer unabhängiger Dev-Datensatz mit zuverlässigen
Belegannotationen bleibt für belastbare Qualitätsaussagen erforderlich.

## Tokens und Laufzeit

Mittelwerte je Fall, CPU auf dieser Maschine. Die vorhandene API-Latenz bleibt
getrennt von der Gesamtlaufzeit. Retrieval-/Embedding-Teilzeiten sind verschachtelt
und dürfen nicht zusätzlich zur Pipelinezeit summiert werden.

| Methode | Durchlauf | Input-Tokens | Output-Tokens | API-Zeit | Gesamtzeit | Retrieval | Query-Embedding |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 (`hybrid`) | Erster Lauf | 7266.1 | 200.7 | 1.673 s | 2.033 s | 0.004 s | 0.000 s |
| BM25 (`hybrid`) | Warme Wiederholung | 7266.1 | 195.6 | 1.411 s | 1.421 s | 0.002 s | 0.000 s |
| BM25 + Vektoren (`hybrid_dense`) | Erster Lauf | 6781.2 | 209.5 | 1.776 s | 14.000 s | 11.856 s | 4.379 s |
| BM25 + Vektoren (`hybrid_dense`) | Warme Wiederholung | 6781.2 | 210.3 | 1.609 s | 5.744 s | 4.125 s | 4.117 s |

Der erste Vektordurchlauf enthält drei neue Dokumentindizes. Alle 50 Fälle der
warmen Wiederholung nutzen Memory-Hits. Beide BM25-Durchläufe nutzen keine
Embeddings. Die BGE-M3-Gewichte waren bereits vorhanden; dieser Lauf enthält
keinen Modelldownload.

| Büchlein | Indexaufbau einschließlich Modellstart | Modellstart darin | Dokument-Encoding darin |
|---|---:|---:|---:|
| DE | 132.156 s | 8.020 s | 124.083 s |
| FR | 114.315 s | 0.000 s | 114.297 s |
| IT | 126.809 s | 0.000 s | 126.745 s |

Die Vektormethode braucht warm im Mittel **5.744 s**, BM25 **1.421 s** pro Fall.
Sie verwendet hier etwas weniger Input-Tokens, benötigt auf CPU aber deutlich
mehr Gesamtzeit. Dies sind lokale Beobachtungen, keine GPU-/MPS-/Server-Prognosen.

## Artefakte und Prüfungen

- [Vollständiger Evaluator-Report](hybrid_quality_2026-10-05.json)
- [Zusätzliche Berechnungen und Integritätsprüfung](hybrid_quality_2026-10-05.summary.json)
- [Originale API-Antworten mit Nutzung und Anfragennummer](hybrid_quality_2026-10-05.api.jsonl)
- [Alle Pipeline-Ergebnisse mit Fallzuordnung](hybrid_quality_2026-10-05.cases.jsonl)
- [Laufumgebung und Versionen](hybrid_quality_environment.json)
- [Ausführbarer Experiment-Harness](../../scripts/run_hybrid_quality_comparison.py)

Das Anfragebudget wurde vor dem Lauf mit kontrollierten Fake-Antworten geprüft:
sowohl erfolgreiche Anfragen als auch fehlgeschlagene Versuche zählen; weitere
Versuche werden vor einem Netzwerkaufruf abgewiesen. Der Harness wurde kompiliert.
Nach dem Lauf wurden 200 erfolgreiche Anfragen, 200 Vorhersagen, vier vollständige
Phasen, gleiche Fallreihenfolgen, nachgerechnetes F1 und unveränderte Pipeline-
Prüfsummen kontrolliert. Die Inference-/Retrieval-Implementierung und Standardwerte
wurden für diesen Vergleich nicht verändert. Es wurden keine zusätzlichen
kostenpflichtigen Benchmarks, Modelltrainings, Commits oder Deployments ausgeführt.
