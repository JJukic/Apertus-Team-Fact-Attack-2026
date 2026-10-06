# Paired E5/title vs BM25/title diagnostics (k=5)

Same frozen dev rows as the local comparison. Changes below concern lexical reference-fragment coverage, not model decisions.

| Language pair | Cases | Coverage improved | Coverage reduced | Mean change (percentage points) |
|---|---|---|---|---|
| de->de | 124 | 28 | 44 | +0.90 |
| de->fr | 136 | 48 | 42 | +1.67 |
| de->it | 106 | 34 | 26 | +6.46 |
| fr->de | 113 | 52 | 25 | +12.49 |
| fr->fr | 140 | 27 | 60 | -7.46 |
| fr->it | 114 | 17 | 52 | -5.07 |
| it->de | 105 | 52 | 19 | +15.79 |
| it->fr | 125 | 32 | 48 | +8.12 |
| it->it | 127 | 21 | 62 | -3.11 |

## Largest coverage gains

- `hf-1149` (de->it, change +100.0 points): Laut der Zusammenfassung gilt der Wolf aufgrund der Überarbeitung der Jagdgesetze nicht länger als geschützte Art.
  BM25 pages: [44, 43, 38, 32, 37]; E5/title pages: [26, 35, 6, 34, 32].
- `hf-1026` (de->it, change +100.0 points): Laut der Zusammenfassung wurden die finanziellen Hilfen auf Personen begrenzt, die bereits genügend Unterstützung erhalten hatten.
  BM25 pages: [45, 30, 9, 37, 1]; E5/title pages: [31, 3, 9, 37, 8].
- `hf-1155` (de->it, change +100.0 points): Laut der Zusammenfassung wurden die finanziellen Hilfen auf Personen begrenzt, die bereits genügend Unterstützung erhalten hatten.
  BM25 pages: [45, 30, 9, 37, 1]; E5/title pages: [31, 3, 9, 37, 8].
- `hf-1100` (de->it, change +100.0 points): Die Zusammenfassung besagt, dass die Maßnahmen der COVID-19-Gesetzgebung bis Ende 2025 verlängert werden.
  BM25 pages: [1, 42, 56, 52, 9]; E5/title pages: [42, 1, 8, 52, 43].
- `hf-0182` (fr->de, change +100.0 points): Le résumé précise que les entreprises suisses doivent contrôler leurs propres activités ainsi que celles de leurs filiales, fournisseurs et partenaires commerciaux, pour veiller au respect, à l’étranger, des normes internationales reconnues en matière de droits de l’homme et d’environnement.
  BM25 pages: [3, 18, 1, 8, 32]; E5/title pages: [8, 4, 18, 3, 9].

## Largest coverage losses

- `hf-1124` (de->de, change -100.0 points): Laut dem Abstimmungstext empfiehlt die Bundesversammlung Volk und Ständen, die Initiative anzunehmen.
  BM25 pages: [1, 31, 30, 22, 7]; E5/title pages: [3, 7, 1, 20, 6].
- `hf-1064` (de->fr, change -100.0 points): Laut der Zusammenfassung ist die Anwendung SwissCovid derzeit aktiv und wurde nicht deaktiviert.
  BM25 pages: [8, 45, 1, 56, 9]; E5/title pages: [52, 42, 1, 51, 53].
- `hf-0270` (fr->de, change -100.0 points): Le résumé précise que la E-ID est facultative.
  BM25 pages: [7, 56, 28, 6, 20]; E5/title pages: [27, 40, 39, 32, 23].
- `hf-1289` (fr->it, change -100.0 points): Le résumé précise que les loups ne sont désormais plus une espèce protégée.
  BM25 pages: [43, 44, 42, 38, 6]; E5/title pages: [26, 35, 36, 32, 37].
- `hf-1412` (it->de, change -100.0 points): Nel riassunto si sostiene che la revoca dell’accordo sulla libera circolazione delle persone non farebbe decadere automaticamente gli altri sei accordi delle Bilaterali I, dato che non è prevista alcuna clausola ghigliottina.
  BM25 pages: [4, 24, 14, 21, 1]; E5/title pages: [14, 2, 24, 1, 22].

These examples were selected after inspecting the measured dev results and are diagnostic, not independent validation. A reduction may reflect omitted relevant text or the removal of irrelevant parts of a long reference section; a gain may include repeated boilerplate. Inspect source text before attributing causes.
