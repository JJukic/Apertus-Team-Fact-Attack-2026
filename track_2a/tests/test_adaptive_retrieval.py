"""Source integrity, non-lossy generators and contradiction-aware ranking."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.evidence_units import EvidenceUnit, EvidenceUnitParser, expand_neighbors, detect_language
from src.adaptive_retrieval import (AdaptiveSettings, BM25Generator, DenseGenerator,
                                    Candidate, candidate_union, NLIReranker, AdaptiveRetriever)
from src.embeddings import HybridSettings
import tempfile


def unit(index, text, *, page=1, section="Council", language="de"):
    return EvidenceUnit("a"*64, page, f"p{page}:b{index}", section, language,
                        text, (0, 0, 10, 10), index)


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_union_retains_both_independent_sets_and_original_source(self):
        units = [unit(i, f"Original passage {i}") for i in range(60)]
        result = candidate_union({"bm25_original": [(u, 2) for u in units[:20]],
                                  "bm25_translated": [(u, 3) for u in units[20:40]],
                                  "dense": [(u, .7) for u in units[40:]]})
        self.assertEqual(len(result), 60)
        self.assertEqual({r.unit.id for r in result}, {u.id for u in units})
        self.assertIs(result[0].unit, units[0])

    def test_union_deduplicates_ids_but_keeps_generator_agreement(self):
        source = unit(0, "Exact original evidence.")
        result = candidate_union({"bm25_original": [(source, 3)], "dense": [(source, .8)]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].ranks, {"bm25_original": 1, "dense": 1})

    def test_contradiction_and_entailment_both_outrank_topic_only(self):
        candidates = [Candidate(unit(i, text)) for i,text in enumerate(("support", "reject", "topic"))]
        reranker = NLIReranker(AdaptiveSettings())
        reranker.score = MagicMock(return_value=[
            {"score": .95, "probabilities": {"entailment": .9, "contradiction": .05, "neutral": .05}, "windows": 1},
            {"score": .96, "probabilities": {"entailment": .01, "contradiction": .95, "neutral": .04}, "windows": 1},
            {"score": .1, "probabilities": {"entailment": .05, "contradiction": .05, "neutral": .9}, "windows": 1}])
        ranked = reranker.rerank("original claim", candidates)
        self.assertEqual([r.unit.text for r in ranked], ["reject", "support", "topic"])
        self.assertEqual(reranker.score.call_args.args[0], "original claim")

    def test_neighbors_do_not_cross_section_or_document_and_keep_pages(self):
        units = [unit(0, "before", page=1), unit(1, "decisive", page=2),
                 unit(2, "qualifier", page=2), unit(3, "other section", section="Other")]
        expanded = expand_neighbors([units[1]], units, radius=2, same_section=True)
        self.assertEqual(expanded, units[:3])
        self.assertEqual([u.page for u in expanded], [1, 2, 2])
        self.assertEqual(expand_neighbors([units[1]], units, 0), [units[1]])
        with self.assertRaises(ValueError):
            expand_neighbors([unit(1, "changed text")], units, 1)

    def test_neighbor_radius_counts_paragraphs_and_retains_cross_section_metadata(self):
        from dataclasses import replace
        units = [unit(0, "Previous condition"), replace(unit(1, "Heading"), kind="heading"),
                 unit(2, "Decisive paragraph"), replace(unit(3, "Heading", section="Next"), kind="heading"),
                 unit(4, "Following qualifier", section="Next")]
        expanded = expand_neighbors([units[2]], units, radius=1)
        self.assertEqual([u.order for u in expanded], [0, 2, 4])
        self.assertEqual(expanded[-1].section, "Next")

    def test_translated_lexical_generator_is_independently_measurable(self):
        units = [unit(0, "Le bailleur doit fournir son autorisation écrite.", language="fr"),
                 unit(1, "Le programme routier prévoit un nouveau tunnel.", language="fr"),
                 unit(2, "Informations générales sur le scrutin.", language="fr")]
        retriever = AdaptiveRetriever(units, AdaptiveSettings(dense=False, ranking="union", neighbor_radius=0))
        result = retriever.retrieve("Eine schriftliche Zustimmung ist notwendig.",
            {"fr": "Le bailleur doit fournir son autorisation écrite."}, "fr")
        self.assertEqual(result["generators"]["bm25_translated"][0]["id"], units[0].id)
        self.assertIn("bm25_original", result["generators"])

    def test_dense_search_uses_cosine_and_preserves_source_ids(self):
        units = [unit(0, "Passage A"), unit(1, "Passage B")]
        backend = MagicMock()
        backend.encode.side_effect = [np.array([[2, 0], [0, 3]]), np.array([[0, 7]])]
        with tempfile.TemporaryDirectory() as folder:
            settings = HybridSettings(cache_dir=Path(folder))
            generator = DenseGenerator(units, settings, backend)
            results = generator.retrieve("query", 2)
            self.assertEqual(results[0][0].id, units[1].id)
            self.assertAlmostEqual(results[0][1], 1)
            self.assertAlmostEqual(results[1][1], 0)

    def test_search_dehyphenation_never_changes_evidence_text(self):
        source = unit(0, "Die Unter-\nmiete benötigt eine schriftliche Zustimmung.")
        generator = BM25Generator([source, unit(1, "Die Nationalstrassen werden ausgebaut.")])
        self.assertEqual(generator.retrieve("Untermiete schriftliche Zustimmung", 1)[0][0], source)
        self.assertIn("Unter-\nmiete", source.text)

    def test_actual_pdf_excludes_known_offpage_heading(self):
        parser = EvidenceUnitParser()
        units = parser.parse(BASE / "data/booklets/2024-11-24_de.pdf", "de")
        wrong_page = [u for u in units if u.page == 33 and "Dritte Vorlage" in u.text]
        self.assertEqual(wrong_page, [])
        self.assertTrue(any(u.page == 34 and "Dritte Vorlage" in u.text for u in units))
        self.assertTrue(all(u.document_id == units[0].document_id for u in units))

    def test_language_detection_for_all_required_languages(self):
        for language, text in {"de": "Der Bundesrat empfiehlt die Annahme der Volksinitiative.",
                               "fr": "Le Conseil fédéral recommande de rejeter cette initiative.",
                               "it": "Il Consiglio federale raccomanda di respingere questa iniziativa."}.items():
            with self.subTest(language=language):
                self.assertEqual(detect_language(text), language)


if __name__ == "__main__":
    unittest.main()
