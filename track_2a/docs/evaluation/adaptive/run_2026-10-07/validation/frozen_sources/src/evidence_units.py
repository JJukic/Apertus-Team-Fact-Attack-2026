"""Layout-derived original evidence blocks with stable, physical-page provenance."""
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import re
import statistics
from typing import Optional

from src.embeddings import pdf_hash

LANGUAGES = ("de", "fr", "it")


@lru_cache(maxsize=1)
def _language_identifier():
    try:
        from langid.langid import LanguageIdentifier, model
    except ImportError as exc:
        raise RuntimeError("Install requirements-adaptive.txt for language detection") from exc
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(list(LANGUAGES))
    return identifier


def detect_language(text: str) -> str:
    if not text.strip():
        raise ValueError("Cannot detect the language of empty text")
    return _language_identifier().classify(text)[0]


@dataclass(frozen=True)
class EvidenceUnit:
    document_id: str
    page: int
    paragraph_id: str
    section: str
    language: str
    text: str
    bbox: tuple
    order: int
    kind: str = "paragraph"

    @property
    def id(self):
        return f"{self.document_id[:16]}:{self.paragraph_id}"

    def to_dict(self):
        return {**asdict(self), "id": self.id, "page_number": self.page}


class EvidenceUnitParser:
    VERSION = "layout-blocks-v2-visible-original-headings-tables"

    def __init__(self):
        self.diagnostics = {}

    def parse(self, path: Path, language: Optional[str] = None):
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("Install requirements-adaptive.txt for layout-aware parsing") from exc
        if language is not None and language not in LANGUAGES:
            raise ValueError("Booklet language must be de, fr or it")
        path = Path(path)
        digest = pdf_hash(path)
        extracted, discarded = [], 0
        with pymupdf.open(path) as document:
            for page_index, page in enumerate(document):
                # Keep MuPDF text blocks, line breaks, lists and source wording.
                blocks = page.get_text("dict", clip=page.rect, sort=True)["blocks"]
                table_boxes = [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
                for block in blocks:
                    if block["type"] != 0:
                        continue
                    lines, sizes, bold_lengths, text_lengths = [], [], [], []
                    for line in block["lines"]:
                        current = []
                        for span in line["spans"]:
                            rectangle = pymupdf.Rect(span["bbox"])
                            visible = (span.get("char_flags", 24) & 24) and span.get("alpha", 255) > 0
                            if not visible or not rectangle.intersects(page.rect):
                                discarded += 1
                                # Do not concatenate across excluded text.
                                if current:
                                    lines.append("".join(current))
                                    current = []
                                lines.append("<excluded-pdf-span>")
                                continue
                            current.append(span["text"])
                            sizes.append(float(span["size"]))
                            text_lengths.append(len(span["text"]))
                            bold_lengths.append(len(span["text"]) if span["flags"] & 16 else 0)
                        if current:
                            lines.append("".join(current))
                    text = "\n".join(lines).strip()
                    if not text or text == "<excluded-pdf-span>":
                        continue
                    # Split at excluded fragments; synthetic markers are never evidence.
                    for part_index, part in enumerate(text.split("<excluded-pdf-span>")):
                        part = part.strip()
                        if not part:
                            continue
                        extracted.append({"page": page_index+1,
                            "paragraph_id": f"p{page_index+1:04d}:b{block['number']:04d}:s{part_index}",
                            "text": part, "sizes": sizes, "bbox": tuple(block["bbox"]),
                            "mostly_bold": sum(bold_lengths) >= .8*max(1, sum(text_lengths)),
                            "table": any(rect.contains(pymupdf.Rect(block["bbox"])) for rect in table_boxes)})
            page_count = len(document)
        if not extracted:
            raise ValueError("PDF contains no visible extractable text; provide OCR with verified provenance")
        detected = detect_language("\n".join(b["text"] for b in extracted)[:12000])
        language = language or detected
        body_sizes = [size for block in extracted for size in block["sizes"] if size > 0]
        median_size = statistics.median(body_sizes) if body_sizes else 10.0
        section, units = "", []
        for block in extracted:
            text = block["text"]
            heading = (len(text) < 240 and not text.strip().isdigit() and
                       ((block["sizes"] and max(block["sizes"]) > median_size*1.15) or
                        block["mostly_bold"] or
                        (len(text) > 8 and text.isupper())))
            if heading:
                section = text
            kind = "table" if block["table"] else "heading" if heading else ("list" if re.match(r"\s*(?:[–•-]|\d+[.)]|[a-z][.)])\s", text)
                                                  else "paragraph")
            units.append(EvidenceUnit(digest, block["page"], block["paragraph_id"], section,
                                      language, text, block["bbox"], len(units), kind))
        self.diagnostics = {"parser_version": self.VERSION, "pymupdf_version": pymupdf.VersionBind,
            "document_sha256": digest, "pages": page_count, "units": len(units),
            "discarded_nonpainting_or_offpage_spans": discarded,
            "detected_language": detected, "declared_language": language,
            "limits": "Text blocks approximate paragraphs. Arbitrary occlusion and scanned tables require visual/OCR review."}
        return units


def expand_neighbors(selected, units, radius=0, *, same_section=False):
    """Expand by nearby body paragraphs; headings do not consume the radius.

    Source section/page fields stay attached even when crossing a heading.
    An explicit same-section variant is available for ablation.
    """
    if type(radius) is not int or radius < 0:
        raise ValueError("Neighbor radius must be a nonnegative integer")
    by_id = {u.id: u for u in units}
    chosen = {}
    for source in selected:
        if source.id not in by_id or source != by_id[source.id]:
            raise ValueError("Selected evidence belongs to a different document")
        chosen[source.id] = source
        eligible = [u for u in units if u.document_id == source.document_id and u.kind != "heading"
                    and (not same_section or u.section == source.section)]
        before = [u for u in eligible if u.order < source.order]
        after = [u for u in eligible if u.order > source.order]
        for neighbor in (before[-radius:] if radius else []) + after[:radius]:
            chosen[neighbor.id] = neighbor
    return sorted(chosen.values(), key=lambda u: u.order)
