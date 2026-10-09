import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import pymupdf

from scripts.analyze_competition_errors import audit, validate_annotations
from src.evidence import SourcePages


class TestCompetitionErrorAudit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        path = self.directory / 'source.pdf'
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((50, 50), 'The proposal concerns biodiversity, not a wage referendum.')
            document.save(path)
        claim = {'text': 'The government accepts a minimum wage referendum.', 'language': 'en'}
        self.cases = {
            'A': {'id': 'A', 'claim': claim, 'vote': 'Biodiversity',
                  'booklet': {'path': 'source.pdf', 'language': 'en'}},
            'B': {'id': 'B', 'claim': claim, 'vote': 'Biodiversity',
                  'reference': {'text': 'The government rejects the biodiversity proposal.', 'language': 'en'}},
        }
        block = SourcePages(self.directory).get(self.cases['A'], 1)['blocks'][0]
        self.annotation = {
            'categories': [], 'review_status': 'source_reviewed',
            'source_reason': 'Rejection of biodiversity does not reject a different wage initiative.',
            'source_proof': {'audit_source_case_id': 'A', 'pdf_path': 'source.pdf', 'page': 1,
                             'pdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'verbatim_text_block': block},
            'reference_sha256': hashlib.sha256(self.cases['B']['reference']['text'].encode()).hexdigest(),
        }

    def test_beginner_review_uses_paired_pdf_and_its_actual_input_reference(self):
        validate_annotations({'B': self.annotation}, self.cases, self.directory)
        changed = copy.deepcopy(self.cases)
        changed['B']['reference']['text'] += ' Changed source.'
        with self.assertRaisesRegex(ValueError, 'Beginner input reference changed'):
            validate_annotations({'B': self.annotation}, changed, self.directory)

    def test_invented_quotation_wrong_page_or_unrelated_source_cannot_confirm_causes(self):
        for field, value in [('verbatim_text_block', 'An invented source quote.'), ('page', 2),
                             ('pdf_sha256', 'old-revision')]:
            annotation = copy.deepcopy(self.annotation)
            annotation['source_proof'][field] = value
            with self.assertRaises(ValueError):
                validate_annotations({'B': annotation}, self.cases, self.directory)
        unrelated = copy.deepcopy(self.cases)
        # Replace the shared object so only the paired source claim changes.
        unrelated['A']['claim'] = {'text': 'Another claim.', 'language': 'en'}
        with self.assertRaisesRegex(ValueError, 'reviewed claim/vote'):
            validate_annotations({'B': self.annotation}, unrelated, self.directory)

    def test_unresolved_scope_does_not_become_a_confirmed_semantic_cause(self):
        annotation = copy.deepcopy(self.annotation)
        annotation.update(review_status='scope_adjudication_pending', categories=['G'])
        predictions = {'A': {'id': 'A', 'label': 1, 'label_name': 'neutral', 'evidence': []},
                       'B': {'id': 'B', 'label': 2, 'label_name': 'contradiction', 'evidence': []}}
        gold = {'A': {'id': 'A', 'label': 1, 'reference': None},
                'B': {'id': 'B', 'label': 1, 'reference': None}}
        result = audit(self.cases, gold, predictions, self.directory,
                       annotations={'B': annotation})
        self.assertEqual(result['categories']['C']['confirmed_count'], 1)
        self.assertEqual(result['categories']['G']['confirmed_count'], 0)
        self.assertEqual(result['semantic_review']['scope_adjudication_pending_ids'], ['B'])
        self.assertEqual(result['semantic_review']['unreviewed_ids'], [])

    def test_translated_run_audits_recorded_context_without_replacing_it_with_baseline(self):
        path = self.directory / 'source.pdf'
        path.unlink()
        with pymupdf.open() as document:
            document.new_page().insert_text((50, 50), 'The proposal concerns biodiversity.')
            document.new_page().insert_text((50, 50), 'The government rejects the minimum wage referendum.')
            document.save(path)
        predictions = {'A': {'id': 'A', 'label': 1, 'label_name': 'neutral', 'evidence': []},
                       'B': {'id': 'B', 'label': 1, 'label_name': 'neutral', 'evidence': []}}
        gold = {'A': {'id': 'A', 'label': 2, 'reference': 'The government rejects the minimum wage referendum.'},
                'B': {'id': 'B', 'label': 1, 'reference': None}}
        record = {'raw_result': {'context_pages': [2], 'retrieval_query_metadata': {'mode': 'translated'}}}
        report = audit(self.cases, gold, predictions, self.directory, records={'A': record})
        self.assertEqual(report['cases'][0]['context_pages'], [2])
        self.assertTrue(report['cases'][0]['retrieval_fuzzy_passage_hit'])
        self.assertIn('recorded_translated', report['cases'][0]['context_provenance'])
        record['raw_result']['context_pages'] = [99]
        with self.assertRaisesRegex(ValueError, 'unique source pages'):
            audit(self.cases, gold, predictions, self.directory, records={'A': record})

    def test_valid_source_proof_cannot_transfer_a_cause_to_changed_predictions(self):
        predictions = {'A': {'id': 'A', 'label': 1, 'label_name': 'neutral', 'evidence': []},
                       'B': {'id': 'B', 'label': 2, 'label_name': 'contradiction', 'evidence': []}}
        gold = {'A': {'id': 'A', 'label': 1, 'reference': None},
                'B': {'id': 'B', 'label': 1, 'reference': None}}
        annotation = copy.deepcopy(self.annotation)
        annotation['applies_to'] = {'gold_label': 1, 'predicted_label': 2, 'context_pages': []}
        audit(self.cases, gold, predictions, self.directory, annotations={'B': annotation})
        predictions['B']['label'] = 0
        predictions['B']['label_name'] = 'entailment'
        with self.assertRaisesRegex(ValueError, 'no longer matches labels'):
            audit(self.cases, gold, predictions, self.directory, annotations={'B': annotation})


if __name__ == '__main__':
    unittest.main()
