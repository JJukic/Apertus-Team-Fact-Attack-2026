# Official offline evaluator

`evaluate.py` is an unmodified copy of the official HackApertus starter evaluator
at commit `5e6957729b77a126006049305ad1ef241211a5b4`:
https://gitlab.com/ifsoftware/hackapertus-starter/-/blob/5e6957729b77a126006049305ad1ef241211a5b4/evaluate.py

`provenance.json` records its source and SHA-256. Keep this file unchanged;
local integration belongs in `src/official_evaluation.py`. Run outside the
prediction container, with gold labels kept separate from prediction input:

```bash
python -m vendor.hackapertus_starter.evaluate \
  --predictions output/predictions.jsonl \
  --expected evaluation/expected-labels.jsonl \
  --cases evaluation/cases.jsonl --json output/score.json
```

The starter's Task B threshold is 0.70. The optimization goal retains the user's
stricter 0.75 acceptance threshold. Token/time averages are self-reported values,
not the competition's relative efficiency scores.
