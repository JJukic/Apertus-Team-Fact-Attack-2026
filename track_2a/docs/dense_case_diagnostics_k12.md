# Paired E5/title vs BM25/title diagnostics (k=12)

Same frozen dev rows as the local comparison. Changes below concern lexical reference-fragment coverage, not model decisions.

| Language pair | Cases | Coverage improved | Coverage reduced | Mean change (percentage points) |
|---|---|---|---|---|
| de->de | 124 | 35 | 28 | +2.07 |
| de->fr | 136 | 41 | 40 | +5.01 |
| de->it | 106 | 25 | 24 | +1.94 |
| fr->de | 113 | 40 | 25 | +6.94 |
| fr->fr | 140 | 35 | 28 | +0.42 |
| fr->it | 114 | 27 | 33 | -1.60 |
| it->de | 105 | 42 | 17 | +11.32 |
| it->fr | 125 | 40 | 28 | +5.43 |
| it->it | 127 | 27 | 33 | -0.17 |

## Largest coverage gains

- `hf-1328` (it->de, change +100.0 points): Il Consiglio federale afferma che il palmoleo potrà essere importato senza restrizioni di sostenibilità e senza dazi doganali.
  BM25 pages: [56, 3, 1, 54, 47, 34, 36, 21, 22, 26, 6, 27]; E5/title pages: [53, 8, 9, 52, 47, 54, 46, 51, 50, 45, 44, 48].
- `hf-1062` (de->de, change +88.8 points): Der Bundesrat spricht sich dafür aus, die Änderung des Jagdgesetzes abzulehnen.
  BM25 pages: [88, 2, 37, 32, 6, 55, 1, 9, 13, 11, 75, 65]; E5/title pages: [7, 26, 38, 32, 6, 44, 88, 37, 36, 35, 40, 42].
- `hf-1188` (fr->de, change +88.8 points): Le Conseil fédéral préconise le rejet de la modification de la loi sur la chasse.
  BM25 pages: [88, 2, 3, 37, 32, 6, 1, 26, 46, 55, 58, 66]; E5/title pages: [26, 7, 38, 6, 32, 44, 88, 37, 35, 36, 40, 42].
- `hf-1279` (fr->de, change +88.8 points): Le Conseil fédéral préconise le rejet de la modification de la loi sur la chasse.
  BM25 pages: [88, 2, 3, 37, 32, 6, 1, 26, 46, 55, 58, 66]; E5/title pages: [26, 7, 38, 6, 32, 44, 88, 37, 35, 36, 40, 42].
- `hf-0380` (it->de, change +79.5 points): Il Consiglio federale propone di approvare l’emendamento alla legge federale sull’imposta federale diretta.
  BM25 pages: [1, 88, 46, 55, 2, 7, 84, 49, 9, 8, 50, 65]; E5/title pages: [46, 55, 9, 56, 8, 49, 3, 88, 1, 58, 2, 54].

## Largest coverage losses

- `hf-1182` (fr->fr, change -65.3 points): Le Conseil fédéral préconise l’acceptation de l’initiative populaire « Davantage de logements abordables ».
  BM25 pages: [17, 16, 1, 32, 5, 8, 11, 4, 15, 3, 12, 14]; E5/title pages: [8, 3, 16, 4, 5, 1, 12, 15, 11, 10, 32, 17].
- `hf-1212` (fr->fr, change -65.3 points): Le Conseil fédéral préconise l’acceptation de l’initiative populaire « Davantage de logements abordables ».
  BM25 pages: [17, 16, 1, 32, 5, 8, 11, 4, 15, 3, 12, 14]; E5/title pages: [8, 3, 16, 4, 5, 1, 12, 15, 11, 10, 32, 17].
- `hf-0194` (fr->fr, change -55.7 points): Le comité affirme que la COMCO préconise l’abolition complète des subventions, qu’elle considère anticonstitutionnelles.
  BM25 pages: [53, 64, 46, 10, 11, 56, 55, 1, 3, 52, 54, 58]; E5/title pages: [46, 53, 50, 10, 63, 56, 55, 11, 51, 61, 60, 3].
- `hf-0219` (fr->fr, change -55.7 points): Le comité affirme que la COMCO préconise l’abolition complète des subventions, qu’elle considère anticonstitutionnelles.
  BM25 pages: [53, 64, 46, 10, 11, 56, 55, 1, 3, 52, 54, 58]; E5/title pages: [46, 53, 50, 10, 63, 56, 55, 11, 51, 61, 60, 3].
- `hf-0329` (fr->fr, change -55.7 points): Le comité affirme que la COMCO préconise l’abolition complète des subventions, qu’elle considère anticonstitutionnelles.
  BM25 pages: [53, 64, 46, 10, 11, 56, 55, 1, 3, 52, 54, 58]; E5/title pages: [46, 53, 50, 10, 63, 56, 55, 11, 51, 61, 60, 3].

These examples were selected after inspecting the measured dev results and are diagnostic, not independent validation. A reduction may reflect omitted relevant text or the removal of irrelevant parts of a long reference section; a gain may include repeated boilerplate. Inspect source text before attributing causes.
