# Erste Fehlerprüfung der vollständigen E6/E7-Validation

Grundlage: 108 eingefrorene Validation-Fälle, E6 Macro-F1 0.7397,
E7 Macro-F1 0.8517. Keine Testfälle wurden für diese Prüfung verwendet.
Die Prüfung verändert weder Goldlabels noch Vorhersagen oder Prompts.

| Fall | Gold | E6 | E7 | Befund aus Originalbelegen und Modellantwort |
|---|---|---|---|---|
| ost-adaptive-0010 | Entailment | Entailment | Neutral | Der überprüfte Goldbeleg auf Seite 26 enthält schriftliche Zustimmung, Unterschrift/qualifizierte Signatur und den Ausschluss einfacher E-Mail. E6 begründet die Unterstützung zutreffend. Die globale Neutral-Schwelle verwirft diese korrekte Antwort. |
| ost-adaptive-0633 | Neutral | Contradiction | Contradiction | Die Modellbegründung sagt selbst, dass ein Einfluss auf das Steuerrecht nicht erwähnt wird, folgert daraus jedoch Widerspruch. Die zitierten Mietrechts-/Gesundheitsabsätze und Seitenzahlen belegen keine Gegenbehauptung. Hier ist die NLI-Entscheidung das sichtbare Problem; Textidentität allein bestätigt keine inhaltliche Belegqualität. |
| ost-adaptive-1457 | Contradiction | Entailment | Entailment | Der italienische Claim vertauscht die Kündigungsfristen: 6 Monate für Wohnungen, 3 für Geschäftsräume. Die gelieferten französischen Originalbelege auf Seiten 8/36 nennen 3 beziehungsweise 6. Die Modellbegründung gibt diese richtigen Quellenwerte wieder, liefert dennoch Entailment. Die erforderlichen Informationen wurden gefunden; Relation/Zahlzuordnung beziehungsweise Labelentscheidung schlägt fehl. |

Die übrigen Fehler sind damit noch nicht manuell kategorisiert. E7 hat 16
Fehler gegenüber den eingefrorenen Goldlabels. Die drei Beispiele zeigen
unterschiedliche Ursachen und dürfen nicht als vollständige Ursachenverteilung
dargestellt werden.

Die Kalibrierung verändert Labels und leert Belege für Neutral. Die ursprüngliche
Modellbegründung bleibt im aktuellen E7-Replay erhalten und kann deshalb
eine Unterstützung beschreiben, obwohl das kalibrierte Label Neutral lautet.
`raw_label`, E6-Originalantwort und `neutral_policy` machen den Vorgang
nachvollziehbar; eine spätere Ausgabe sollte rohe Modellbegründung und
Kalibrierungsentscheidung klar unterscheiden. Dieser Lauf wird dafür nicht
nachträglich verändert.

Weitere Prüfungen nach dem kontrollierten Vergleich: Neutral versus fehlende
Information, explizite Zuordnung von Zahlen zu Entitäten, Ausschluss allein
stehender Seitenzahlen als entscheidende Belege und Stabilität der globalen
Neutral-Schwelle. Erst die übrigen Ablationen zeigen, welche Varianten diese
Fehler bereits reduzieren.
