from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.prepare_competition_cases import build_folds
from src.measurement import interval_union_seconds, RequestRecorder


class TestCompetitionMeasurement(unittest.TestCase):
    def test_union_counts_overlapping_calls_once_and_clips_boundaries(self):
        self.assertEqual(interval_union_seconds([(0, 5), (3, 8)], 0, 10), 8)
        self.assertEqual(interval_union_seconds([(-2, 2), (1, 4), (7, 15)], 0, 10), 7)
        self.assertEqual(interval_union_seconds([(10, 11), (-3, -1)], 0, 10), 0)
        self.assertEqual(interval_union_seconds([], 0, 10), 0)

    def test_all_attempts_recorded_with_unknown_failed_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestRecorder(Path(directory) / 'requests.jsonl', 'test')
            def failing(**kwargs):
                raise TimeoutError('do not store exception bodies')
            with recorder.case('case-A'), self.assertRaises(TimeoutError):
                recorder.wrap(failing)(model='apertus', messages=[])
            event = json.loads(recorder.journal.read_text())
            self.assertEqual(event['case_id'], 'case-A')
            self.assertEqual(event['status'], 'error')
            self.assertFalse(event['usage_known'])
            self.assertIsNone(event['input_tokens'])
            self.assertNotIn('do not store exception bodies', recorder.journal.read_text())

    def test_concurrent_case_ids_and_uncapped_api_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestRecorder(Path(directory) / 'requests.jsonl', 'test')
            def create(**kwargs):
                self.assertNotIn('max_tokens', kwargs)
                self.assertNotIn('max_completion_tokens', kwargs)
                return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                                       choices=[SimpleNamespace(message=SimpleNamespace(content='{}'),
                                                               finish_reason='stop')])
            call = recorder.wrap(create)
            def worker(cid):
                with recorder.case(cid):
                    call(model='apertus', messages=[], max_tokens=160, max_completion_tokens=120)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker, ['A', 'B']))
            self.assertEqual({e['case_id'] for e in recorder.events}, {'A', 'B'})
            self.assertEqual(sum(e['input_tokens'] for e in recorder.events), 6)
            self.assertTrue(all(e['usage_known'] for e in recorder.events))

    def test_date_folds_keep_all_languages_and_tasks_together(self):
        rows = [{'booklet_publish_date': date, 'reference_language': language}
                for date in ('2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01')
                for language in ('de', 'fr', 'it')]
        folds = build_folds(rows, n_folds=2)
        self.assertEqual(folds, build_folds(list(reversed(rows)), n_folds=2))
        self.assertEqual(len(folds), 4)
        self.assertEqual(set(folds.values()), {0, 1})


if __name__ == '__main__':
    unittest.main()
