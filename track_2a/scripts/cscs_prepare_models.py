"""Fetch immutable retrieval models and require offline query-cache hits.

Run inside the CSCS Python environment before submitting GPU preparation.
No Apertus credentials are needed or used by this helper.
"""
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src import config
from src.adaptive_retrieval import AdaptiveSettings
from src.grounded_apertus import QueryPreparer
from huggingface_hub import snapshot_download


class OfflineTransport:
    model = config.LLM_NAME

    def request(self, *args, **kwargs):
        raise RuntimeError("A faithful query is missing; prepare translations locally before submitting GPU work")


def main():
    preparer = QueryPreparer(OfflineTransport(), BASE/".cache/adaptive/queries")
    cases = [json.loads(line) for line in (BASE/"data/evaluation/adaptive/validation.jsonl").read_text().splitlines()]
    for case in cases:
        preparer.prepare(case["claim"],case["claim_language"],
                         translate=case["claim_language"] != case["booklet_language"])
    print(f"Verified offline query preparation for {len(cases)} cases.", flush=True)
    settings = AdaptiveSettings()
    for model, revision in [(config.EMBEDDING_MODEL, config.EMBEDDING_REVISION),
                            (settings.reranker_model, settings.reranker_revision)]:
        path = snapshot_download(model, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "README.md"],
            max_workers=2)
        print(json.dumps({"model":model,"revision":revision,"snapshot":path}),flush=True)


if __name__ == "__main__":
    main()
