"""
PDF Parser for Swiss Official Voting Booklets (Abstimmungsbüchlein).
Extracts full text and structured paragraphs with page attribution.

Key feature: DYNAMIC proposal boundary detection — works for any booklet,
any voting date, any number of proposals. No hardcoded page numbers.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import pypdf


# PRIMARY: Numbered voting object headings — these appear exactly once per proposal intro page
# and are the most reliable signal across all Swiss booklet languages
_PRIMARY_PROPOSAL_PATTERNS = [
    re.compile(r"\b(?:erste|zweite|dritte|vierte|fünfte)\s+vorlage\b", re.IGNORECASE),
    re.compile(r"\b(?:premier|deuxi[eè]me|troisi[eè]me|quatri[eè]me)\s+objet\b", re.IGNORECASE),
    re.compile(r"\b(?:primo|secondo|terzo|quarto)\s+oggetto\b", re.IGNORECASE),
]

# FALLBACK: Generic title patterns used only if PRIMARY yields no results
_FALLBACK_PROPOSAL_PATTERNS = [
    re.compile(r"\bvolksinitiative\b", re.IGNORECASE),
    re.compile(r"\b(?:bundesgesetz|bundesbeschluss)\s+(?:über|zur|zum|betreffend)\b", re.IGNORECASE),
    re.compile(r"\bänderung\s+(?:des|der)\s+\w*gesetzes\b", re.IGNORECASE),
    re.compile(r"\binitiative\s+populaire\b", re.IGNORECASE),
    re.compile(r"\b(?:loi|arrêté)\s+fédéral[e]?\s+(?:sur|concernant|relatif|du)\b", re.IGNORECASE),
    re.compile(r"\biniziativa\s+popolare\b", re.IGNORECASE),
    re.compile(r"\b(?:legge|decreto)\s+federale\s+(?:sulla|sul|concernente|del)\b", re.IGNORECASE),
]

# Signals that a page is a "summary/key-fact" page for a proposal
_SUMMARY_PAGE_PATTERNS = [
    # German
    r"empfehlung\s+von\s+bundesrat",
    r"abstimmungsfrage",
    r"darum\s+geht\s+es",
    r"auf\s+einen\s+blick",
    # French
    r"recommandation\s+du\s+conseil\s+fédéral",
    r"question\s+(?:soumise|votée)",
    r"l'essentiel\s+en\s+bref",
    # Italian
    r"raccomandazione\s+del\s+consiglio\s+federale",
    r"quesito\s+(?:posto\s+)?in\s+votazione",
    r"in\s+breve",
]

_COMPILED_SUMMARY = [re.compile(p, re.IGNORECASE) for p in _SUMMARY_PAGE_PATTERNS]

# Table-of-contents indicators — pages before the actual content
_TOC_PATTERNS = [
    re.compile(r"inhalt|inhaltsverzeichnis|sommaire|indice|table\s+des\s+mati", re.IGNORECASE),
]


class PDFParser:
    # Increment when paragraph extraction or source metadata changes.
    VERSION = "paragraphs-v1-min40"

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

    # ------------------------------------------------------------------
    # Dynamic proposal detection
    # ------------------------------------------------------------------

    def _detect_proposal_boundaries(self, pages: List[Dict[str, Any]]) -> Tuple[Dict[int, int], List[int]]:
        """
        Dynamically detect which page starts which proposal (1-indexed).
        Returns: ({page_number: proposal_id}, [proposal_start_pages])

        Strategy:
        - Scan the FIRST 200 chars of each page for a title-level proposal heading.
        - Enforce a minimum page gap between proposals (≥ 8 pages) to avoid
          picking up sub-section headings inside a single proposal.
        - Pages before the first proposal get proposal_id = 0 (TOC/cover).
        """
        MIN_PROPOSAL_GAP = 8  # A new proposal can't start within 8 pages of the last
        # Ordinal patterns: map each language ordinal to a proposal number (1-based)
        ORDINAL_EXTRACT = [
            (re.compile(r"\berste\s+vorlage\b", re.IGNORECASE), 1),
            (re.compile(r"\bzweite\s+vorlage\b", re.IGNORECASE), 2),
            (re.compile(r"\bdritte\s+vorlage\b", re.IGNORECASE), 3),
            (re.compile(r"\bvierte\s+vorlage\b", re.IGNORECASE), 4),
            (re.compile(r"\bfünfte\s+vorlage\b", re.IGNORECASE), 5),
            (re.compile(r"\bsechste\s+vorlage\b", re.IGNORECASE), 6),
            (re.compile(r"\bpremier\s+objet\b", re.IGNORECASE), 1),
            (re.compile(r"\bdeuxi[eè]me\s+objet\b", re.IGNORECASE), 2),
            (re.compile(r"\btroisi[eè]me\s+objet\b", re.IGNORECASE), 3),
            (re.compile(r"\bquatri[eè]me\s+objet\b", re.IGNORECASE), 4),
            (re.compile(r"\bcinqui[eè]me\s+objet\b", re.IGNORECASE), 5),
            (re.compile(r"\bsixi[eè]me\s+objet\b", re.IGNORECASE), 6),
            (re.compile(r"\bprimo\s+oggetto\b", re.IGNORECASE), 1),
            (re.compile(r"\bsecondo\s+oggetto\b", re.IGNORECASE), 2),
            (re.compile(r"\bterzo\s+oggetto\b", re.IGNORECASE), 3),
            (re.compile(r"\bquarto\s+oggetto\b", re.IGNORECASE), 4),
            (re.compile(r"\bquinto\s+oggetto\b", re.IGNORECASE), 5),
            (re.compile(r"\bsesto\s+oggetto\b", re.IGNORECASE), 6),
        ]

        seen_ordinals: set = set()
        ordinal_to_page: Dict[int, int] = {}
        proposal_starts: List[int] = []

        # Patterns that indicate a voting-text sub-section or final ballot page (NOT a new proposal)
        SUBPAGE_EXCLUSIONS = [
            re.compile(r"abstimmungstext", re.IGNORECASE),
            re.compile(r"texte\s+soumis\s+au\s+vote", re.IGNORECASE),
            re.compile(r"testo\s+(?:in\s+votazione|sottoposto)", re.IGNORECASE),
            re.compile(r"bundesbeschluss\s+zur\s+volksinitiative", re.IGNORECASE),
            re.compile(r"arr[eê]t[eé]\s+f[eé]d[eé]ral\s+relatif", re.IGNORECASE),
            re.compile(r"decreto\s+federale\s+concernente", re.IGNORECASE),
        ]

        total_pages = max((p["page_number"] for p in pages), default=1)
        late_page_threshold = int(total_pages * 0.88)

        ordinal_all_pages: Dict[int, List[int]] = {}  # ordinal -> all pages it appeared on

        for page in pages:
            pg = page["page_number"]
            title_zone = page["text"][:200]

            # Skip cover/TOC pages
            if pg <= 3:
                continue

            # Skip very late pages (likely index/ballot/closing)
            if pg >= late_page_threshold and len(ordinal_all_pages) >= 2:
                continue

            # Skip sub-section pages
            is_subpage = any(exc.search(title_zone) for exc in SUBPAGE_EXCLUSIONS)
            if is_subpage:
                continue

            for pattern, ordinal in ORDINAL_EXTRACT:
                if pattern.search(title_zone):
                    ordinal_all_pages.setdefault(ordinal, []).append(pg)
                    break

        # For each ordinal, prefer the SECOND occurrence if it's ≥8 pages
        # after the first (first = overview page, second = real section start).
        # Otherwise, use the first valid occurrence.
        for ordinal, pgs in ordinal_all_pages.items():
            if len(pgs) >= 2 and (pgs[1] - pgs[0]) >= 8:
                ordinal_to_page[ordinal] = pgs[1]
            else:
                ordinal_to_page[ordinal] = pgs[0]

        if ordinal_to_page:
            # Build sorted list but enforce minimum gap (overview pages can be
            # only 1-2 pages apart — the real section boundary should be farther)
            sorted_pages = [ordinal_to_page[k] for k in sorted(ordinal_to_page.keys())]
            proposal_starts = [sorted_pages[0]]
            for pg in sorted_pages[1:]:
                if pg - proposal_starts[-1] >= 8:
                    proposal_starts.append(pg)
        else:
            # Fallback: generic patterns with gap enforcement
            for page in pages:
                pg = page["page_number"]
                title_zone = page["text"][:300]
                is_subpage = any(exc.search(title_zone) for exc in SUBPAGE_EXCLUSIONS)
                if is_subpage:
                    continue
                hit = any(p.search(title_zone) for p in _FALLBACK_PROPOSAL_PATTERNS)
                if hit and pg > 3 and (not proposal_starts or (pg - proposal_starts[-1]) >= MIN_PROPOSAL_GAP):
                    proposal_starts.append(pg)

        # Build page → proposal_id mapping
        page_to_proposal: Dict[int, int] = {}
        for page in pages:
            pg = page["page_number"]
            proposal_id = 0
            for i, start_pg in enumerate(proposal_starts):
                if pg >= start_pg:
                    proposal_id = i + 1  # 1-indexed
            page_to_proposal[pg] = proposal_id

        return page_to_proposal, proposal_starts

    def _is_summary_page(self, page_text: str) -> bool:
        """Return True if this page is a key-facts/summary page for a proposal."""
        snippet = page_text[:600]
        return any(p.search(snippet) for p in _COMPILED_SUMMARY)

    def _is_toc_page(self, page_text: str, page_number: int) -> bool:
        """Return True if this looks like a table of contents or cover page."""
        if page_number > 4:
            return False
        return any(p.search(page_text) for p in _TOC_PATTERNS)

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_paragraphs(self, pdf_path: Union[str, Path], min_length: int = 40) -> List[Dict[str, Any]]:
        """
        Segment the booklet into paragraphs with page attribution.
        Uses DYNAMIC proposal boundary detection — no hardcoded page numbers.
        """
        pages = self.extract_pages(pdf_path)
        page_to_proposal, proposal_starts = self._detect_proposal_boundaries(pages)

        paragraphs = []
        para_id = 0

        for page in pages:
            pg = page["page_number"]
            page_text = page["text"]
            proposal_id = page_to_proposal.get(pg, 0)

            # Determine section type
            if self._is_toc_page(page_text, pg):
                section_type = "toc"
            elif self._is_summary_page(page_text):
                section_type = "summary"
            elif proposal_id == 0:
                section_type = "cover"
            else:
                # Rough sub-section within a proposal based on relative position
                prop_start = proposal_starts[proposal_id - 1] if proposal_id <= len(proposal_starts) else pg
                rel_page = pg - prop_start
                if rel_page <= 3:
                    section_type = "intro"
                elif rel_page <= 8:
                    section_type = "detail"
                elif rel_page <= 13:
                    section_type = "arguments"
                else:
                    section_type = "legal_text"

            raw_chunks = page_text.split("\n\n")
            for chunk in raw_chunks:
                clean_chunk = " ".join(chunk.split()).strip()
                if len(clean_chunk) >= min_length:
                    paragraphs.append({
                        "id": para_id,
                        "page_number": pg,
                        "proposal_id": proposal_id,
                        "section_type": section_type,
                        "text": clean_chunk,
                    })
                    para_id += 1

        return paragraphs
