"""
Synthetic Benchmark Generator for Track 2A (OST).
Generates challenging, document-grounded test claims from historical Swiss voting booklets
across DE, FR, IT for all three NLI classes (Entailment, Neutral, Contradiction).
Used to rigorously test pipeline generalization and verify zero overfitting.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.apertus_client import ApertusClient
from src.pdf_parser import PDFParser
from src import config

logger = logging.getLogger(__name__)


def generate_claims_for_context(
    client: ApertusClient,
    context: str,
    proposal_title: str,
    lang: str = "de",
) -> List[Dict[str, Any]]:
    """
    Call Apertus to create 1 Entailment, 1 Neutral, and 1 Contradiction claim for a given passage.
    """
    lang_instructions = {
        "de": "Formuliere alle Behauptungen auf Deutsch.",
        "fr": "Formulez toutes les affirmations en français.",
        "it": "Formula tutte le affermazioni in italiano.",
    }
    lang_instruction = lang_instructions.get(lang, lang_instructions["de"])

    prompt = f"""Du bist ein NLI-Benchmark-Kurator für offizielle Schweizer Abstimmungsunterlagen.
Vorlage: {proposal_title}

Erstelle basierend auf dem folgenden Textabschnitt genau drei präzise, faktenbasierte Behauptungen:
1. ENTAILMENT (Label 0): Eine Behauptung, die durch den Text eindeutig und zwingend wahr / belegt ist.
2. NEUTRAL (Label 1): Eine Behauptung zum selben Thema, die plausibel klingt, deren Wahrheitsgehalt aber im Text weder belegt noch widerlegt wird (nicht im Text erwähnt).
3. CONTRADICTION (Label 2): Eine Behauptung, die einer konkreten Tatsache, Zahl oder Empfehlung im Text direkt widerspricht.

{lang_instruction}

Textabschnitt:
\"\"\"{context}\"\"\"

Antworte ausschliesslich mit einem gültigen JSON-Array von 3 Objekten im folgenden Format:
[
  {{"claim": "...", "entailment_label": 0, "reasoning": "..."}},
  {{"claim": "...", "entailment_label": 1, "reasoning": "..."}},
  {{"claim": "...", "entailment_label": 2, "reasoning": "..."}}
]
"""
    try:
        response = client.client.chat.completions.create(
            model=client.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        # Clean markdown code blocks if present
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # Parse JSON array directly using boundary brackets
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                items = json.loads(content[start : end + 1])
                if isinstance(items, list):
                    return items
            except Exception:
                pass

        # Non-greedy regex fallback
        match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        if match:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                return items
    except Exception as e:
        logger.error(f"Error generating claims: {e}")
    return []


def create_benchmark_from_booklet(
    booklet_date: str = "2024-11-24",
    output_path: Optional[Path] = None,
    samples_per_proposal: int = 1,
) -> Path:
    """
    Extracts representative passages from each proposal of the booklet in DE, FR, IT,
    generates 3 claims (0, 1, 2) per passage, and writes to a JSONL benchmark file.
    """
    output_file = output_path or (config.DATA_DIR / f"benchmark_{booklet_date}.jsonl")
    client = ApertusClient()
    parser = PDFParser()

    all_samples = []

    languages = ["de", "fr", "it"]
    for lang in languages:
        pdf_path = config.BOOKLETS_DIR / f"{booklet_date}_{lang}.pdf"
        if not pdf_path.exists():
            print(f"Skipping {pdf_path} (not found)")
            continue

        paragraphs = parser.extract_paragraphs(pdf_path)
        # Find unique proposal IDs
        prop_ids = sorted(list(set(p.get("proposal_id", 0) for p in paragraphs if p.get("proposal_id", 0) > 0)))

        print(f"\nProcessing {booklet_date}_{lang}.pdf: {len(prop_ids)} proposals found: {prop_ids}")

        for pid in prop_ids:
            # Pick the top informative paragraphs for this proposal (length > 250, not TOC)
            candidates = [
                p for p in paragraphs
                if p.get("proposal_id") == pid and len(p.get("text", "")) > 250 and p.get("page_number", 0) > 4
            ]
            if not candidates:
                continue

            selected = candidates[:samples_per_proposal]
            for c in selected:
                context_text = c["text"]
                generated = generate_claims_for_context(
                    client=client,
                    context=context_text,
                    proposal_title=f"Vorlage {pid}",
                    lang=lang,
                )

                for item in generated:
                    sample = {
                        "claim": item["claim"],
                        "claim_language": lang,
                        "entailment_label": int(item["entailment_label"]),
                        "reference_string": context_text,
                        "booklet_date": booklet_date,
                        "booklet_language": lang,
                        "reasoning": item.get("reasoning", ""),
                        "proposal_id": pid,
                        "page_number": c.get("page_number"),
                    }
                    all_samples.append(sample)
                    print(f"  [{lang.upper()} Prop {pid} Label {sample['entailment_label']}]: {sample['claim'][:60]}...")

    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_samples)} synthetic benchmark claims saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    create_benchmark_from_booklet("2024-11-24")
