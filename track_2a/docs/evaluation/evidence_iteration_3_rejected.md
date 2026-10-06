# Iteration 3: vollständige Zuordnung aller Modellzitate

Status: **verworfen, ausschließlich Offline-Experiment**. Keine Änderung in
`verify_claim`, keine neue Modellantwort oder API-Anfrage.

Ausgangspunkt ist Replay V2 mit normalisiertem Zitatabgleich und erneuter
Zahlenprüfung im zugeordneten Zitat. Die einzige experimentelle Änderung in
V3: Entailment wird zu Neutral, sobald mindestens ein Modellzitat nicht
wörtlich zugeordnet werden kann. Bei Neutral werden Belege geleert.

| Methode / Durchlauf | V2 Macro-F1 | V3 Macro-F1 | Neue richtige Labels gegenüber V2 | Richtige Labels verloren gegenüber V2 |
|---|---:|---:|---:|---:|
| hybrid / first_pass | 0,6410 | 0,5368 | 0 | 5 |
| hybrid / warm_repeat | 0,6412 | 0,5554 | 0 | 4 |
| hybrid_dense / first_pass | 0,7194 | 0,6776 | 0 | 2 |
| hybrid_dense / warm_repeat | 0,7396 | 0,6985 | 0 | 2 |

Betroffene Fälle gegenüber V2:

- Hybrid erster Durchlauf: `historical-00`, `historical-12`, `historical-18`,
  `historical-29`, `cross-de-it-0` (jeweils richtiges Entailment zu Neutral).
- Hybrid Wiederholung: `historical-00`, `historical-12`, `historical-18`,
  `cross-de-it-0` (jeweils richtiges Entailment zu Neutral).
- Hybrid Dense beide Durchläufe: `historical-18`, `historical-20`
  (Gold-Label Entailment zu Neutral) und `historical-27`
  (falsches Entailment zu ebenfalls falschem Neutral bei Gold Contradiction).

Die problematische Teilzuordnung von `historical-20` wird verhindert. Die Regel
ist jedoch zu streng für Antworten mit einem hinreichenden wörtlichen Beleg und
einem zusätzlichen umformulierten Zitat. Zudem kann auch ein vollständig
wörtliches Zitat sachfremd sein; die Regel garantiert keine semantische
Beleggültigkeit. Sie besteht daher weder die Score- noch die Qualitätsbedingung.

Fallresultate, Werte nach Sprache und Konfusionsmatrizen stehen in
`evidence_replay_2026-10-06_v3.{json,csv,md}`. Für V1/V2/V3 sind dieselben 50
Entwicklungsfälle, Gold-Labels, PDFs, gespeicherten Antworten und ausgewählten
Seiten eingefroren. Keine Aussage über unabhängige Testdaten.

Reproduktion aus `track_2a`:

```sh
../.venv/bin/python scripts/replay_evidence_comparison.py --version v3
```

Nächster Versuch: Ein expliziter, optionaler Promptzusatz verlangt kurze
wörtliche Passagen, die den vollständigen Claim belegen, und eine konsistente
Beziehung zwischen Begründung und Label. Er ist als
`candidate_evidence_prompt_v1.txt` eingefroren. Ohne neue Modellantworten lässt
sich sein Effekt nicht durch Replay bewerten; bis zu einem kontrollierten
Live-Vergleich ist er kein bestätigter Gewinn.
