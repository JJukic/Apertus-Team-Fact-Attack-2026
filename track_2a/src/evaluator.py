"""
Benchmark Evaluator for Track 2A (OST).
Calculates Macro-F1 across Entailment (0), Neutral (1), and Contradiction (2),
breaks down metrics by language (DE, FR, IT), and tracks token and latency efficiency.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.metrics import classification_report, f1_score
from tabulate import tabulate
from tqdm import tqdm

from src.inference import ClaimVerificationEngine, PredictionResult
from src import config

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    def __init__(self, engine: Optional[ClaimVerificationEngine] = None):
        self.engine = engine or ClaimVerificationEngine()

    def evaluate(
        self,
        dataset_path: Optional[Path] = None,
        strategy: str = "retrieval",
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = Path(dataset_path) if dataset_path else config.BENCHMARK_PATH
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if limit:
            records = records[:limit]

        print(f"\nRunning benchmark evaluation on {len(records)} samples using strategy '{strategy}'...")

        y_true = []
        y_pred = []
        languages = []
        latencies = []
        prompt_tokens = []
        total_tokens = []
        completion_tokens = []
        language_pairs = []
        pipeline_latencies = []
        retrieval_stats = []
        page_recalls = []
        results = []
        evidence_hits = []

        import re

        for record in tqdm(records, desc="Evaluating"):
            claim = record["claim"]
            true_label = int(record["entailment_label"])
            claim_lang = record.get("claim_language", "de")
            booklet_lang = record.get("booklet_language", claim_lang)
            requested_booklet_lang = booklet_lang
            gold_reference = record.get("reference_string", "").strip()

            # Map to local booklet PDF
            booklet_date = record.get("booklet_date", "2026-06-14")
            pdf_path = config.BOOKLETS_DIR / f"{booklet_date}_{booklet_lang}.pdf"
            if not pdf_path.exists():
                # Fallback to German booklet of same date if specific language is missing
                pdf_path = config.BOOKLETS_DIR / f"{booklet_date}_de.pdf"
                logger.warning("Missing %s booklet; evaluating actual language pair %s->de", booklet_lang, claim_lang)
                booklet_lang = "de"

            vote = record.get("vote")
            if isinstance(vote, dict):
                vote = vote.get("title") or vote.get("text")

            pred: PredictionResult = self.engine.verify_claim(
                claim=claim,
                booklet_pdf=pdf_path,
                claim_language=claim_lang,
                strategy=strategy,
                vote=vote,
            )

            # Evaluate evidence match against gold reference string
            evidence_matched = False
            if gold_reference and pred.evidence:
                gold_words = set(re.findall(r"\w+", gold_reference.lower()))
                for ev in pred.evidence:
                    ev_words = set(re.findall(r"\w+", ev.lower()))
                    if not ev_words or not gold_words:
                        continue
                    overlap = len(gold_words & ev_words) / min(len(ev_words), len(gold_words))
                    if overlap >= 0.30 or ev.lower() in gold_reference.lower() or gold_reference.lower() in ev.lower():
                        evidence_matched = True
                        break
            if gold_reference:
                evidence_hits.append(1 if evidence_matched else 0)

            y_true.append(true_label)
            y_pred.append(pred.label)
            languages.append(claim_lang)
            latencies.append(pred.latency_ms)
            prompt_tokens.append(pred.tokens_prompt)
            total_tokens.append(pred.tokens_total)
            completion_tokens.append(pred.tokens_completion)
            pipeline_latencies.append(pred.total_latency_ms)
            retrieval_stats.append(pred.retrieval_metrics)
            language_pairs.append(f"{claim_lang}->{booklet_lang}")

            # Only explicit, physical 1-based gold pages qualify; never infer
            # them from word overlap or proposal IDs. Existing data has none.
            gold_pages = record.get("gold_evidence_pages")
            page_recall = None
            if (booklet_lang == requested_booklet_lang and isinstance(gold_pages, list) and gold_pages
                    and all(type(p) is int and p > 0 for p in gold_pages)):
                predicted_pages = {s.page_number for s in pred.evidence_sources if s.page_number is not None}
                page_recall = len(set(gold_pages) & predicted_pages) / len(set(gold_pages))
                page_recalls.append(page_recall)

            results.append({
                "claim": claim,
                "claim_lang": claim_lang,
                "booklet_lang": booklet_lang,
                "requested_booklet_lang": requested_booklet_lang,
                "language_pair": language_pairs[-1],
                "true_label": true_label,
                "pred_label": pred.label,
                "reasoning": pred.reasoning,
                "evidence": pred.evidence,
                "gold_reference": gold_reference[:120] + "..." if len(gold_reference) > 120 else gold_reference,
                "evidence_matched": evidence_matched,
                "latency_ms": pred.latency_ms,
                "tokens": pred.tokens_total,
                "input_tokens": pred.tokens_prompt,
                "output_tokens": pred.tokens_completion,
                "total_latency_ms": pred.total_latency_ms,
                "retrieval_metrics": pred.retrieval_metrics,
                "actual_strategy": pred.strategy,
                "evidence_page_recall": page_recall,
            })

        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0
        avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0
        evidence_alignment = sum(evidence_hits) / len(evidence_hits) if evidence_hits else 1.0

        # Language breakdown
        lang_metrics = {}
        for l in sorted(set(languages)):
            sub_true = [yt for yt, lang in zip(y_true, languages) if lang == l]
            sub_pred = [yp for yp, lang in zip(y_pred, languages) if lang == l]
            lang_f1 = f1_score(sub_true, sub_pred, average="macro", labels=[0, 1, 2], zero_division=0)
            lang_metrics[l] = {
                "count": len(sub_true),
                "macro_f1": round(lang_f1, 4),
            }

        pair_metrics = {}
        for pair in sorted(set(language_pairs)):
            indices = [i for i, value in enumerate(language_pairs) if value == pair]
            pair_metrics[pair] = {
                "count": len(indices),
                "macro_f1": round(f1_score([y_true[i] for i in indices], [y_pred[i] for i in indices],
                                          average="macro", labels=[0, 1, 2], zero_division=0), 4),
            }
        cross_indices = [i for i, pair in enumerate(language_pairs) if pair.split("->")[0] != pair.split("->")[1]]
        cross_f1 = (round(f1_score([y_true[i] for i in cross_indices], [y_pred[i] for i in cross_indices],
                                  average="macro", labels=[0, 1, 2], zero_division=0), 4)
                    if cross_indices else None)
        def average(values):
            return round(sum(values) / len(values), 2) if values else 0.0

        report = {
            "strategy": strategy,
            "sample_count": len(records),
            "macro_f1": round(macro_f1, 4),
            "evidence_alignment_rate": round(evidence_alignment, 4),
            "reference_string_count": len(evidence_hits),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_total_tokens": round(avg_total_tokens, 1),
            "by_language": lang_metrics,
            "by_language_pair": pair_metrics,
            "cross_language_macro_f1": cross_f1,
            "cross_language_sample_count": len(cross_indices),
            "mock_mode": bool(self.engine.client.mock),
            "evidence_alignment_method": "Approximate lexical word overlap >= 0.30 / substring; not semantic quality",
            "evidence_page_recall": average(page_recalls) if page_recalls else None,
            "gold_page_sample_count": len(page_recalls),
            "avg_output_tokens": average(completion_tokens),
            "avg_total_latency_ms": average(pipeline_latencies),
            "cache_status_counts": dict(Counter(m.get("cache_status", "not_used") for m in retrieval_stats)),
            "actual_strategy_counts": dict(Counter(r["actual_strategy"] for r in results)),
            "retrieval_timing_ms": {
                name: average([m.get(name, 0.0) for m in retrieval_stats])
                for name in ("document_preparation_ms", "retrieval_ms", "embedding_ms", "model_load_ms",
                             "index_build_ms", "index_load_ms", "document_embedding_ms", "query_embedding_ms")
            },
            "results": results,
            "detailed_classification": classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=["0: Entailment", "1: Neutral", "2: Contradiction"],
                output_dict=True,
                zero_division=0,
            ),
        }

        self.print_report(report)
        return report

    @staticmethod
    def print_report(report: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("           TRACK 2A (OST) BENCHMARK REPORT")
        print("=" * 60)
        print(f"Strategy:            {report['strategy']}")
        print(f"Total Samples:       {report['sample_count']}")
        print(f"PRIMARY METRIC:      Macro-F1 = {report['macro_f1']:.4f}")
        if report.get("mock_mode"):
            print("MOCK RUN:            No real Apertus quality/latency conclusions")
        if report.get("reference_string_count", 0) > 0:
            print(f"Evidence Alignment:  {report['evidence_alignment_rate'] * 100:.1f}% ({report['reference_string_count']} references; lexical approximation)")
        print(f"Avg Input Tokens:    {report['avg_prompt_tokens']}")
        print(f"Avg Total Tokens:    {report['avg_total_tokens']}")
        print(f"Avg Latency:         {report['avg_latency_ms']} ms")
        print(f"Avg Output Tokens:   {report.get('avg_output_tokens', 0)}")
        print(f"Avg Pipeline Time:   {report.get('avg_total_latency_ms', 0)} ms (API latency above unchanged)")
        print(f"Cross-language F1:   {report.get('cross_language_macro_f1')} (None = no cross-language samples)")
        print(f"Cache Statuses:      {report.get('cache_status_counts', {})}")
        print("-" * 60)

        # Classification Table
        cls_data = []
        det = report["detailed_classification"]
        for lbl in ["0: Entailment", "1: Neutral", "2: Contradiction"]:
            d = det.get(lbl, {})
            cls_data.append([
                lbl,
                f"{d.get('precision', 0):.2f}",
                f"{d.get('recall', 0):.2f}",
                f"{d.get('f1-score', 0):.2f}",
                d.get("support", 0),
            ])
        print("\nClassification Performance:")
        print(tabulate(cls_data, headers=["Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="github"))

        # Language Breakdown Table
        print("\nLanguage Breakdown:")
        lang_data = []
        for lang, vals in report["by_language"].items():
            lang_data.append([lang.upper(), vals["count"], f"{vals['macro_f1']:.4f}"])
        print(tabulate(lang_data, headers=["Language", "Samples", "Macro-F1"], tablefmt="github"))
        if report.get("by_language_pair"):
            print("\nClaim -> Actual Booklet Language:")
            print(tabulate([[pair, vals["count"], vals["macro_f1"]]
                            for pair, vals in report["by_language_pair"].items()],
                           headers=["Language Pair", "Samples", "Macro-F1"], tablefmt="github"))
        print("=" * 60 + "\n")
