"""
Benchmark Evaluator for Track 2A (OST).
Calculates Macro-F1 across Entailment (0), Neutral (1), and Contradiction (2),
breaks down metrics by language (DE, FR, IT), and tracks token and latency efficiency.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from sklearn.metrics import classification_report, f1_score
from tabulate import tabulate
from tqdm import tqdm

from src.inference import ClaimVerificationEngine, PredictionResult
from src import config


class BenchmarkEvaluator:
    def __init__(self, engine: Optional[ClaimVerificationEngine] = None):
        self.engine = engine or ClaimVerificationEngine()

    def evaluate(
        self,
        dataset_path: Optional[Path] = None,
        strategy: str = "retrieval",
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = dataset_path or config.BENCHMARK_PATH
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
        results = []

        for record in tqdm(records, desc="Evaluating"):
            claim = record["claim"]
            true_label = int(record["entailment_label"])
            claim_lang = record.get("claim_language", "de")
            booklet_lang = record.get("booklet_language", claim_lang)

            # Map to local booklet PDF
            pdf_path = config.BOOKLETS_DIR / f"2026-06-14_{booklet_lang}.pdf"
            if not pdf_path.exists():
                # Fallback to German booklet if specific language is missing
                pdf_path = config.BOOKLETS_DIR / "2026-06-14_de.pdf"

            pred: PredictionResult = self.engine.verify_claim(
                claim=claim,
                booklet_pdf=pdf_path,
                claim_language=claim_lang,
                strategy=strategy,
            )

            y_true.append(true_label)
            y_pred.append(pred.label)
            languages.append(claim_lang)
            latencies.append(pred.latency_ms)
            prompt_tokens.append(pred.tokens_prompt)
            total_tokens.append(pred.tokens_total)

            results.append({
                "claim": claim,
                "claim_lang": claim_lang,
                "true_label": true_label,
                "pred_label": pred.label,
                "reasoning": pred.reasoning,
                "evidence": pred.evidence,
                "latency_ms": pred.latency_ms,
                "tokens": pred.tokens_total,
            })

        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0
        avg_total_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0

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

        report = {
            "strategy": strategy,
            "sample_count": len(records),
            "macro_f1": round(macro_f1, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_total_tokens": round(avg_total_tokens, 1),
            "by_language": lang_metrics,
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
        print(f"Avg Input Tokens:    {report['avg_prompt_tokens']}")
        print(f"Avg Total Tokens:    {report['avg_total_tokens']}")
        print(f"Avg Latency:         {report['avg_latency_ms']} ms")
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
        print("=" * 60 + "\n")
