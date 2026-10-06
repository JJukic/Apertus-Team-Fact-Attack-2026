"""Regression checks using visually reviewed historical source pages."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from check_page_attribution import bounded_pages
from src.embeddings import pdf_hash
from src.evidence_matcher import match_quote
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, RectangleObject


class SyntheticAttributionTests(unittest.TestCase):
    def extract(self, content, cropbox=None):
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(content.encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        if cropbox:
            page.cropbox = RectangleObject(cropbox)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pdf"
            with path.open("wb") as handle:
                writer.write(handle)
            return bounded_pages(path)

    def test_offpage_text_does_not_create_an_original_quote(self):
        pages, removed = self.extract(
            "BT /F1 12 Tf 1 0 0 1 20 100 Tm (Visible original sentence.) Tj ET "
            "BT /F1 12 Tf 1 0 0 1 -300 100 Tm (Hidden original sentence.) Tj ET")
        self.assertIsNotNone(match_quote("Visible original sentence.", pages))
        self.assertIsNone(match_quote("Hidden original sentence.", pages))
        self.assertTrue(removed)

    def test_user_transform_and_cropbox_are_both_applied(self):
        pages, _ = self.extract(
            "q 1 0 0 1 -250 0 cm BT /F1 12 Tf 1 0 0 1 270 100 Tm "
            "(Transformed visible passage.) Tj ET Q "
            "BT /F1 12 Tf 1 0 0 1 150 100 Tm (Outside cropped area.) Tj ET",
            cropbox=[0, 0, 100, 200])
        self.assertIsNotNone(match_quote("Transformed visible passage.", pages))
        self.assertIsNone(match_quote("Outside cropped area.", pages))

    def test_dropped_fragment_cannot_stitch_an_evidence_quote(self):
        pages, _ = self.extract(
            "BT /F1 12 Tf 1 0 0 1 20 120 Tm (The claim is ) Tj ET "
            "BT /F1 12 Tf 1 0 0 1 -300 110 Tm (not supported) Tj ET "
            "BT /F1 12 Tf 1 0 0 1 20 100 Tm (supported by this document.) Tj ET")
        self.assertIsNone(match_quote("The claim is supported by this document.", pages))

    def test_nonpainting_modes_and_graphics_state_restore(self):
        pages, _ = self.extract(
            "q BT /F1 12 Tf 3 Tr 1 0 0 1 20 100 Tm (Invisible but in-page sentence.) Tj ET Q "
            "q BT /F1 12 Tf 7 Tr 1 0 0 1 20 90 Tm (Clipping-only hidden sentence.) Tj ET Q "
            "BT /F1 12 Tf 1 0 0 1 20 80 Tm (Visible restored graphic state.) Tj ET")
        self.assertIsNone(match_quote("Invisible but in-page sentence.", pages))
        self.assertIsNone(match_quote("Clipping-only hidden sentence.", pages))
        self.assertIsNotNone(match_quote("Visible restored graphic state.", pages))


class PageAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = BASE / "docs/evaluation"
        cls.plan = json.loads((folder / "evidence_prompt_dev_plan_2026-10-06.json").read_text())
        cls.pages = {}
        for path, digest in cls.plan["pdf_sha256"].items():
            if pdf_hash(Path(path)) != digest:
                raise ValueError("Historical source PDF changed")
            cls.pages[path] = bounded_pages(path)[0]
        cls.results = json.loads((folder / "evidence_prompt_dev_live_2026-10-06.json").read_text())["results"]

    def test_subletting_request_is_on_visible_page_32_not_33(self):
        case = next(r for r in self.results if r["method"] == "hybrid_dense" and r["id"] == "cross-fr-de-2")
        source = case["candidate"]["prediction"]["evidence_sources"][0]
        pages = self.pages[case["candidate"]["prediction"]["booklet_path"]]
        self.assertEqual(source["page_number"], 33)  # Recorded faulty text-layer attribution.
        self.assertIsNone(match_quote(source["quote"], [p for p in pages if p["page_number"] == 33]))
        corrected = match_quote(source["quote"], [p for p in pages if p["page_number"] == 32])
        self.assertIsNotNone(corrected)

    def test_all_dense_quotes_other_than_offpage_duplicates_are_preserved(self):
        snapshots = {(s["method"], s["record"]["id"]): s for s in self.plan["snapshots"]}
        changed = []
        for row in self.results:
            if row["method"] != "hybrid_dense":
                continue
            snapshot = snapshots[row["method"], row["id"]]
            pages = self.pages[snapshot["pdf"]]
            for variant in ("baseline", "candidate"):
                for source in row[variant]["prediction"]["evidence_sources"]:
                    attributed = [p for p in pages if p["page_number"] == source["page_number"]]
                    if not match_quote(source["quote"], attributed):
                        selected = [p for p in pages if p["page_number"] in snapshot["prepared"]["metrics"]["selected_pages"]]
                        corrected = match_quote(source["quote"], selected)
                        self.assertIsNotNone(corrected, (row["id"], variant))
                        self.assertEqual(corrected[0]["page_number"], 32)
                        changed.append((row["id"], variant))
        self.assertCountEqual(changed, [
            ("historical-05", "baseline"), ("historical-05", "candidate"),
            ("historical-26", "candidate"), ("cross-fr-de-2", "baseline"),
            ("cross-fr-de-2", "candidate"), ("cross-it-fr-2", "baseline"),
            ("cross-it-fr-2", "candidate")])


if __name__ == "__main__":
    unittest.main()
