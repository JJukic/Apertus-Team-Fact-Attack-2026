"""Replay recorded responses through current validation without network calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.apertus_client import ApertusClient, ApertusResponseError
from src.embeddings import pdf_hash
from historical_apertus_client import ApertusClient as HistoricalClient
from run_prompt_comparison import summarize


def replay(response):
    """Transport is a local function returning only the stored response."""
    choices = [SimpleNamespace(message=SimpleNamespace(content=c["message"].get("content")),
                               finish_reason=c.get("finish_reason")) for c in response["choices"]]
    stored = SimpleNamespace(choices=choices,
                             usage=SimpleNamespace(**response["usage"]) if response.get("usage") else None)
    client = ApertusClient(mock=True)
    client.mock = False
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: stored)))
    return client.infer("Recorded response validation only", "No new prediction")


def main():
    folder = BASE / "docs/evaluation"
    stem = "evidence_prompt_dev_live_2026-10-06"
    journal = folder / (stem + ".api.jsonl")
    result_path = folder / (stem + ".json")
    result = json.loads(result_path.read_text())
    events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    invalid, unchanged = [], 0
    for event in events:
        response = event["response"]
        try:
            output = replay(response)
        except ApertusResponseError as error:
            invalid.append({"method": event["method"], "variant": event["variant"],
                            "case_id": event["case_id"], "request_number": event["request_number"],
                            "finish_reason": response["choices"][0].get("finish_reason"),
                            "error_type": type(error).__name__,
                            "tokens_prompt": error.tokens_prompt,
                            "tokens_completion": error.tokens_completion})
            continue
        parsed = HistoricalClient._parse_json(response["choices"][0]["message"]["content"])
        label, rule = HistoricalClient._apply_calibrated_decision(
            parsed["p_entail"], parsed["p_neutral"], parsed["p_contra"], parsed["label"])
        if output.label != label or output.decision_rule_applied != rule or output.evidence != parsed["evidence"]:
            raise ValueError("A successful response changed under operational validation")
        unchanged += 1
    excluded_ids = {r["case_id"] for r in invalid}
    common_rows = [r for r in result["results"] if r["id"] not in excluded_ids]
    payload = {"kind": "offline_response_validity_audit", "new_api_requests": 0,
               "recorded_requests": len(events), "valid_responses_unchanged": unchanged,
               "invalid_responses": invalid, "excluded_ids_all_methods": sorted(excluded_ids),
               "full_50_case_comparison_valid": not invalid,
               "common_complete_subset_diagnostic": {m: summarize([r for r in common_rows if r["method"] == m])
                                                      for m in ("retrieval", "hybrid_dense")},
               "subset_note": "Incomplete cases omitted across all methods: diagnostic only, not a replacement full-score or generalization proof.",
               "input_sha256": {p.name: pdf_hash(p) for p in
                   (journal, result_path, BASE / "src/apertus_client.py", Path(__file__))}}
    output_path = folder / "response_validity_iteration_6.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"valid_unchanged": unchanged, "invalid": invalid,
                      "common_subset_case_count_per_method": len(common_rows) // 2}, indent=2))


if __name__ == "__main__":
    main()
