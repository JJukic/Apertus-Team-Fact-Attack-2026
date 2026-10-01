"""
Download script for Track 2A (OST) challenge data:
- Benchmark dataset from Hugging Face (OSTswiss/MNLIoverSwissVotingBooklets)
- Official Swiss Federal Voting Booklets (Abstimmungsbüchlein) from bk.admin.ch (DE, FR, IT)
"""

import os
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOOKLETS_DIR = DATA_DIR / "booklets"

DATASET_URL = (
    "https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets/raw/main/demo_2026_09_28_1130.jsonl"
)

BOOKLETS = {
    "2026-06-14": {
        "de": "https://www.bk.admin.ch/dam/de/sd-web/WeUrKyC0FyPc/2026-06-14_erlaeuterungen_des_bundesrates.pdf",
        "fr": "https://www.bk.admin.ch/dam/fr/sd-web/WeUrKyC0FyPc/2026-06-14_explications_du_conseil_federal.pdf",
        "it": "https://www.bk.admin.ch/dam/it/sd-web/WeUrKyC0FyPc/2026-06-14_spiegazioni_del_consigliofederale.pdf",
    },
    "2024-11-24": {
        "de": "https://www.bk.admin.ch/dam/de/sd-web/wHXT9BH3VlYQ/2024-11-24_erlaeuterungen_des_bundesrates.pdf",
        "fr": "https://www.bk.admin.ch/dam/fr/sd-web/wHXT9BH3VlYQ/2024-11-24_explications_du_conseil_federal.pdf",
        "it": "https://www.bk.admin.ch/dam/it/sd-web/wHXT9BH3VlYQ/2024-11-24_spiegazioni_del_consigliofederale.pdf",
    },
    "2024-09-22": {
        "de": "https://www.bk.admin.ch/dam/de/sd-web/iWYLyEPfLvhn/2024-09-22_erlaeuterungen_des_bundesrates.pdf",
        "fr": "https://www.bk.admin.ch/dam/fr/sd-web/iWYLyEPfLvhn/2024-09-22_explications_du_conseil_federal.pdf",
        "it": "https://www.bk.admin.ch/dam/it/sd-web/iWYLyEPfLvhn/2024-09-22_spiegazioni_del_consigliofederale.pdf",
    },
}


def download_file(url: str, dest_path: Path, desc: str):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [ALREADY EXISTS] {desc}: {dest_path} ({dest_path.stat().st_size / 1024:.1f} KB)")
        return dest_path

    print(f"  [DOWNLOADING] {desc} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "HackApertus-Track2A-Client/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        data = resp.read()
        out_file.write(data)
    print(f"  [SAVED] {desc} -> {dest_path} ({len(data) / 1024:.1f} KB)")
    return dest_path


def main():
    print("=== Downloading Hack Apertus Track 2A (OST) Assets ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BOOKLETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download benchmark dataset
    dataset_dest = DATA_DIR / "demo_dataset.jsonl"
    download_file(DATASET_URL, dataset_dest, "HF Benchmark Dataset")

    # 2. Download official voting booklets (current + historical)
    for date_str, lang_dict in BOOKLETS.items():
        for lang, url in lang_dict.items():
            pdf_dest = BOOKLETS_DIR / f"{date_str}_{lang}.pdf"
            download_file(url, pdf_dest, f"Booklet PDF {date_str} ({lang.upper()})")

    # Check total size
    total_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"\nAll assets ready in {DATA_DIR}!")
    print(f"Total data folder size: {total_size / (1024 * 1024):.2f} MB (Challenge limit: < 100 MB)\n")


if __name__ == "__main__":
    main()
