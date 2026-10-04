"""
Official OST benchmark (Hugging Face: OSTswiss/MNLIoverSwissVotingBooklets).

- Downloads the full dataset (parquet) and every referenced voting booklet PDF.
- Builds a deterministic dev/test split *by voting date*, so that test booklets
  (in all three languages) are never seen during development. This simulates the
  held-out benchmark, which uses different claims and possibly different booklets.

Usage:
    python -m src.hf_dataset            # download data + booklets, write splits
"""

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from src import config

PARQUET_URL = (
    "https://huggingface.co/api/datasets/OSTswiss/MNLIoverSwissVotingBooklets"
    "/parquet/default/train/0.parquet"
)

HF_DIR = config.DATA_DIR / "hf"
PARQUET_PATH = HF_DIR / "train.parquet"
DEV_PATH = HF_DIR / "dev.jsonl"
TEST_PATH = HF_DIR / "test.jsonl"

SPLIT_SEED = 42
TEST_DATE_FRACTION = 0.25
# Booklets the pipeline was already tuned on must never land in the test split.
DEV_PINNED_DATES = {"2026-06-14", "2024-11-24"}


def booklet_filename(url: str, publish_date: str) -> str:
    """'https://www.bk.admin.ch/dam/fr/sd-web/.../x.pdf' + '2026-06-14' -> '2026-06-14_fr.pdf'"""
    m = re.search(r"/dam/(de|fr|it)/", url)
    lang = m.group(1) if m else Path(url).stem[-2:]
    return f"{publish_date}_{lang}.pdf"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HackApertus-Track2A-Client/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  [WARNING] Could not download {url}: {e}")
        return False


def load_rows() -> List[Dict]:
    import pandas as pd

    if not _download(PARQUET_URL, PARQUET_PATH):
        raise RuntimeError("Could not download the OST benchmark parquet file.")
    df = pd.read_parquet(PARQUET_PATH)
    rows = []
    for idx, r in df.iterrows():
        date = str(r["booklet_publish_date"])[:10]
        rows.append({
            "id": f"hf-{idx:04d}",
            "claim": r["claim"],
            "claim_language": r["claim_language"],
            "reference_string": r["reference_string"],
            "reference_language": r["reference_language"],
            "entailment_label": int(r["entailment_label"]),
            "vote": r["vote"],
            "booklet_publish_date": date,
            "booklet_url": r["booklet_url"],
            "booklet_file": booklet_filename(r["booklet_url"], date),
            "baseline_score": float(r["baseline_score"]),
        })
    return rows


def split_dates(dates: List[str], seed: int = SPLIT_SEED, test_fraction: float = TEST_DATE_FRACTION) -> List[str]:
    """Deterministically choose the voting dates that form the test split."""
    unique = sorted(set(dates) - DEV_PINNED_DATES)
    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(set(dates)) * test_fraction))
    return sorted(shuffled[:n_test])


def download_booklets(rows: List[Dict]) -> Dict[str, bool]:
    urls = {r["booklet_file"]: r["booklet_url"] for r in rows}
    status = {}
    for i, (fname, url) in enumerate(sorted(urls.items()), 1):
        ok = _download(url, config.BOOKLETS_DIR / fname)
        status[fname] = ok
        print(f"  [{i}/{len(urls)}] {'OK ' if ok else 'ERR'} {fname}")
    return status


def build_splits(rows: Optional[List[Dict]] = None) -> Dict[str, int]:
    rows = rows if rows is not None else load_rows()
    test_dates = set(split_dates([r["booklet_publish_date"] for r in rows]))
    HF_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"dev": 0, "test": 0}
    with open(DEV_PATH, "w", encoding="utf-8") as dev_f, open(TEST_PATH, "w", encoding="utf-8") as test_f:
        for r in rows:
            split = "test" if r["booklet_publish_date"] in test_dates else "dev"
            (test_f if split == "test" else dev_f).write(json.dumps({**r, "split": split}, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(f"  Test dates ({len(test_dates)}): {', '.join(sorted(test_dates))}")
    return counts


def main():
    print("=== OST benchmark: dataset, booklets, dev/test split ===")
    rows = load_rows()
    print(f"  {len(rows)} rows loaded")
    status = download_booklets(rows)
    missing = [f for f, ok in status.items() if not ok]
    counts = build_splits(rows)
    print(f"  dev={counts['dev']}  test={counts['test']}  -> {HF_DIR}")
    if missing:
        print(f"  [WARNING] {len(missing)} booklets missing: {missing}")


if __name__ == "__main__":
    main()
