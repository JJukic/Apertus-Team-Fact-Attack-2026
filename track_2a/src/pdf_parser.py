"""
PDF Parser for Swiss Official Voting Booklets (Abstimmungsbüchlein).
Extracts full text and structured paragraphs with page attribution.
"""

from pathlib import Path
from typing import List, Dict, Any, Union
import pypdf


class PDFParser:
    def __init__(self, cache_dir: Union[Path, None] = None):
        self.cache_dir = cache_dir

    def extract_pages(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Extract text page by page from the booklet PDF.
        Returns a list of dicts with 'page_number' and 'text'.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "page_number": idx + 1,
                    "text": text,
                })
        return pages

    def extract_full_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract the complete text of the booklet joined by newlines.
        """
        pages = self.extract_pages(pdf_path)
        return "\n\n".join([f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages])

    def extract_paragraphs(self, pdf_path: Union[str, Path], min_length: int = 40) -> List[Dict[str, Any]]:
        """
        Segment the booklet into paragraphs with page attribution.
        """
        pages = self.extract_pages(pdf_path)
        paragraphs = []
        para_id = 0

        for page in pages:
            raw_text = page["text"]
            # Split by double newlines or single newlines followed by capital/bullet
            raw_chunks = raw_text.split("\n\n")
            for chunk in raw_chunks:
                clean_chunk = " ".join(chunk.split()).strip()
                if len(clean_chunk) >= min_length:
                    paragraphs.append({
                        "id": para_id,
                        "page_number": page["page_number"],
                        "text": clean_chunk,
                    })
                    para_id += 1

        return paragraphs
