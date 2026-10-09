from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import config
from src.apertus_client import ApertusClient, NLIOutput
from src.inference import ClaimVerificationEngine
from src.query_translation import QueryTranslator, guard, numbers
from src.retriever import PassageRetriever


class FakeClient(ApertusClient):
    def __init__(self, translation='Le taux normal de TVA passera de 7,7 à 8,1 %.'):
        super().__init__(mock=True)
        self.translation = translation
        self.translation_calls, self.claims = 0, []

    def translate(self, text, target_lang):
        self.translation_calls += 1
        return self.translation, 4, 5, 10

    def infer(self, context, claim, **kwargs):
        self.claims.append(claim)
        return NLIOutput(label=1, reasoning='fixture', p_neutral=1,
                         tokens_prompt=40, tokens_completion=15, tokens_total=55)


class TestQueryTranslation(unittest.TestCase):
    def test_numeric_and_negation_guards_preserve_original_on_changed_meaning(self):
        self.assertIsNone(guard('Der Satz ist nicht 7,7%.', "Le taux n’est pas de 7,7 %.", 'de', 'fr'))
        self.assertEqual(guard('Der Satz ist nicht 7,7%.', 'Le taux est de 7,7%.', 'de', 'fr'), 'negation_presence_changed')
        self.assertEqual(guard('Der Satz beträgt 7,7%.', 'Le taux est de 8,1%.', 'de', 'fr'), 'numeric_values_changed')
        self.assertEqual(numbers('10 000 personnes, 7,7%'), numbers("10’000 Personen, 7.7%"))

    def test_translation_cache_is_single_flight_and_contains_no_labels(self):
        client = FakeClient('Le taux est de 7,7 %.')
        with tempfile.TemporaryDirectory() as directory:
            translator = QueryTranslator(client, directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: translator.get('Der Satz beträgt 7,7%.', 'de', 'fr'), range(16)))
            self.assertEqual(client.translation_calls, 1)
            self.assertEqual(sum(r[1] for r in results), 4)
            self.assertEqual(sum(r[2] for r in results), 5)
            reloaded = QueryTranslator(client, directory).get('Der Satz beträgt 7,7%.', 'de', 'fr')
            self.assertTrue(reloaded[3]['cache_hit'])
            self.assertEqual(client.translation_calls, 1)
            text = next(Path(directory).glob('*.json')).read_text()
            self.assertNotIn('"label"', text)
            self.assertNotIn('reference', text)

    def test_retrieval_translation_never_changes_nli_claim_and_same_language_skips_calls(self):
        corpus = [{'page_number': 1, 'proposal_id': 1, 'text': 'Le taux normal de TVA passera de 7,7 à 8,1 %.'},
                  {'page_number': 2, 'proposal_id': 1, 'text': 'Des recettes supplémentaires seront versées.'},
                  {'page_number': 3, 'proposal_id': 2, 'text': 'Une autre proposition sur le logement.'}]
        client = FakeClient()
        engine = ClaimVerificationEngine(apertus_client=client)
        data = {'paragraphs': corpus, 'pages': corpus, 'retriever': PassageRetriever(corpus)}
        claim = 'Der normale Mehrwertsteuersatz steigt von 7,7 auf 8,1 %.'
        with tempfile.TemporaryDirectory() as directory, patch.object(engine, '_get_booklet_data', return_value=data), \
             patch.object(config, 'QUERY_TRANSLATION_CACHE_DIR', Path(directory)), \
             patch.object(config, 'RETRIEVAL_QUERY_MODE', 'union'), patch.object(config, 'TRANSLATE_CLAIM', False):
            first = engine.verify_claim(claim, 'fixture.pdf', claim_language='de', booklet_language='fr')
            second = engine.verify_claim(claim, 'fixture.pdf', claim_language='de', booklet_language='fr')
            same = engine.verify_claim(claim, 'fixture.pdf', claim_language='de', booklet_language='de')
        self.assertEqual(client.claims, [claim, claim, claim])
        self.assertEqual(client.translation_calls, 1)
        self.assertEqual(first.tokens_total, 64)
        self.assertEqual(second.tokens_total, 55)
        self.assertTrue(first.retrieval_query_metadata['translation_used'])
        self.assertTrue(second.retrieval_query_metadata['cache_hit'])
        self.assertFalse(same.retrieval_query_metadata['translation_used'])

    def test_union_keeps_baseline_scores_and_speaker_pages_without_mutating_source(self):
        corpus = [{'page_number': 1, 'text': 'apple apple tree', 'section': 'committee'},
                  {'page_number': 2, 'text': 'pomme pomme arbre', 'section': 'government'},
                  {'page_number': 3, 'text': 'other text', 'section': 'committee'}]
        retriever = PassageRetriever(corpus)
        old = retriever.retrieve_hybrid('apple', top_k=3)
        new = retriever.retrieve_hybrid('apple', top_k=3, query_variants=['pomme'])
        scores = {p['page_number']: p['retrieval_score'] for p in new}
        self.assertTrue(all(scores[p['page_number']] >= p['retrieval_score'] for p in old))
        self.assertTrue(all('retrieval_score' not in p for p in corpus))
        kept = retriever.retrieve_hybrid('pomme', top_k=2, ensure_sections={'committee': 1}, query_variants=['apple'])
        self.assertTrue(any(p['section'] == 'committee' for p in kept))
        self.assertEqual(len(kept), 2)


if __name__ == '__main__':
    unittest.main()
