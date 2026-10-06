# Offline analysis of baseline errors

Source: `20261006T140501_advanced_hybrid_ids_Apertus-v1.5-70B-thinking_final_docker.json`, commit `e9f6cd7`. Existing live results only; no new model calls.

- 402 cases, 24 misclassifications, 0 technical errors.
- Saved macro-F1: 0.9402.
- Cross-lingual: 18/275 errors; same-language: 6/127.
- 21 distinct claim/vote/language-pair groups among the 24 errors.
- 6 errors mention automatic top-passage fallback; 12 cases overall use it.

## Wrong-label transitions

| Expected | Predicted | Count |
|---|---|---|
| 0 | 1 | 5 |
| 0 | 2 | 7 |
| 1 | 0 | 1 |
| 1 | 2 | 2 |
| 2 | 0 | 2 |
| 2 | 1 | 7 |

## Language pairs (include denominators)

| Pair | Errors / cases | Error rate |
|---|---|---|
| de->de | 2/38 | 5.3% |
| de->fr | 2/48 | 4.2% |
| de->it | 4/46 | 8.7% |
| fr->de | 6/36 | 16.7% |
| fr->fr | 2/56 | 3.6% |
| fr->it | 0/38 | 0.0% |
| it->de | 3/43 | 7.0% |
| it->fr | 3/64 | 4.7% |
| it->it | 2/33 | 6.1% |

## Repeated claims

- `hf-0063, hf-0141`: Wird die Abstimmung angenommen, müssen die Betreiber grosser Wasserkraftwerke gegen Entschädigung genügend Wasser in ihren Anlagen zurückhalten, um in den kalten Monaten Strom zu produzieren.
- `hf-0359, hf-0428`: Se il voto viene approvato, i gestori delle grandi centrali idroelettriche saranno tenuti, in cambio di un indennizzo, a conservare nei propri impianti acqua sufficiente per produrre elettricità nei mesi freddi.
- `hf-0386, hf-0454`: Se la votazione verrà approvata, l’aliquota ordinaria dell’IVA salirà dal 7,7% all’8,1%.

## Interpretation and next measurement

Observed topics include hydropower reserves, pesticide imports, VAT changes and actor-attributed claims. These are review candidates, not established failure causes.
A recorded decision rule does not establish that the rule changed a correct raw label. Original raw responses and selected context are not fully present in this report.
Neutral outputs have no evidence by design: absence of evidence alone is not proof of a retrieval failure.
We cannot decide whether errors come from missing context, language interpretation, source attribution, judgment rules or gold labels without inspecting original context and references.
The diagnostic JSON retains all 24 rows, including repeated claims, with source provenance. It is NOT an independent evaluation set: these are exposed test cases selected because baseline failed. Do not tune on these and then claim held-out improvement.
Use a fixed dev sample with both errors and correct controls for the actual comparison. Score the two methods on identical data and report technical failures separately. The saved baseline predates our stricter response/evidence validation, so rerun the current ids baseline alongside two_stage.

## Offline pipeline coverage

Automated tests exercise a real PDF, retrieval, simulated extraction and judgment responses, source-id mapping, official export, token accounting and stage failures. No network requests are made. These tests establish pipeline behavior, not model accuracy or CSCS latency.
