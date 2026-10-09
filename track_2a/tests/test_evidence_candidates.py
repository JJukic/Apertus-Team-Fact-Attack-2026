import copy
import unittest

from scripts.evaluate_evidence_candidates import construct_evidence


class Pages:
    def __init__(self, pages):
        self.pages = pages

    def get(self, case, number):
        return self.pages.get(number) if type(number) is int and number > 0 else None


class TestEvidenceCandidates(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'source-case', 'booklet': {'path': 'source.pdf'},
                     'claim': {'text': 'Der Bundesrat empfiehlt Nein.'}, 'vote': 'Vorlage'}
        self.prediction = {'id': 'source-case', 'label': 0, 'label_name': 'entailment',
                           'evidence': [{'page': 43, 'text': 'old clipped text'}]}

    def test_verbatim_quote_correct_page_and_no_label_mutation(self):
        text = '2050: «Le Conseil fédéral recommande le rejet.»\nConsiglio federale.'
        original = copy.deepcopy(self.prediction)
        result = construct_evidence(self.prediction, self.case, Pages({43: {'text': text, 'blocks': []}}), 'raw_pages')
        self.assertEqual(result['evidence'], [{'page': 43, 'text': text}])
        self.assertEqual(result['label'], 0)
        self.assertEqual(self.prediction, original)

    def test_first_five_order_and_duplicate_pages(self):
        self.prediction['evidence'] = [{'page': page, 'text': 'old'} for page in [2, 2, 3, 4, 5, 6, 7]]
        source = Pages({page: {'text': f'Source on page {page}', 'blocks': []} for page in range(2, 8)})
        result = construct_evidence(self.prediction, self.case, source, 'raw_pages')
        self.assertEqual([r['page'] for r in result['evidence']], [2, 3, 4, 5, 6])

    def test_neutral_evidence_is_empty(self):
        self.prediction['label'] = 1
        result = construct_evidence(self.prediction, self.case, Pages({}), 'raw_pages')
        self.assertEqual(result['evidence'], [])

    def test_missing_and_invalid_page_metadata_never_become_page_one(self):
        for page in (None, 0, -1, True, '43', 999):
            with self.subTest(page=page):
                self.prediction['evidence'] = [{'page': page, 'text': 'unknown'}]
                result = construct_evidence(self.prediction, self.case, Pages({}), 'raw_pages')
                self.assertEqual(result['evidence'], [])
                self.assertEqual(result['label'], 0)

    def test_maximum_after_unicode_normalization_and_block_fallback(self):
        block = 'Der Bundesrat empfiehlt die Vorlage abzulehnen. ' * 3
        pages = Pages({43: {'text': 'ﬃ' * 1700, 'blocks': [block]}})
        self.assertEqual(construct_evidence(self.prediction, self.case, pages, 'raw_pages')['evidence'], [])
        result = construct_evidence(self.prediction, self.case, pages, 'raw_pages_and_blocks')
        self.assertEqual(result['evidence'], [{'page': 43, 'text': block}])


if __name__ == '__main__':
    unittest.main()
