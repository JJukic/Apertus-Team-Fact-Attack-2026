import copy
import unittest

from src.competition_selection import distribution, quality_component, simulated_score, known_component_pareto, rank_by_known_quality


def summary(f1=.9, hit=.5, tokens=100):
    return {'qualifies': True, 'tasks': {
        'A': {'macro_f1': f1, 'evidence_hit5': hit, 'input_tokens_reported': tokens, 'output_tokens_reported': 10},
        'B': {'macro_f1': .98, 'input_tokens_reported': tokens, 'output_tokens_reported': 10}}}


class TestCompetitionSelection(unittest.TestCase):
    def test_user_weighted_quality_and_explicit_efficiency_assumptions(self):
        baseline = summary()
        expected = .30 * .9 + .12 * .5 + .28 * .98
        self.assertAlmostEqual(quality_component(baseline), expected)
        self.assertAlmostEqual(simulated_score(baseline, baseline, 1), expected + .30)
        self.assertAlmostEqual(simulated_score(baseline, baseline, .5), expected + .15)
        slower = simulated_score(baseline, baseline, 1, (2, 1))
        self.assertAlmostEqual(slower, expected + .30 - .045)
        for reference, factors in [(0, (1, 1)), (1.1, (1, 1)), (.5, (0, 1))]:
            with self.assertRaises(ValueError):
                simulated_score(baseline, baseline, reference, factors)

    def test_frontier_preserves_quality_token_tradeoffs_and_excludes_invalid_or_diagnostic(self):
        baseline, better, expensive = summary(), summary(hit=.8), summary(f1=.95, hit=.85, tokens=200)
        invalid = summary(f1=1, hit=1)
        invalid['qualifies'] = False
        diagnostic = summary(f1=1, hit=1)
        diagnostic['selection_eligible'] = False
        runs = {'baseline': baseline, 'better': better, 'expensive': expensive, 'invalid': invalid, 'diagnostic': diagnostic}
        self.assertEqual(set(known_component_pareto(runs)), {'better', 'expensive'})
        self.assertEqual(rank_by_known_quality(runs), ['expensive', 'better', 'baseline'])

    def test_distribution_does_not_invent_missing_measurements(self):
        self.assertIsNone(distribution([])['p95'])
        self.assertEqual(distribution([1, 10, 2])['median'], 2)


if __name__ == '__main__':
    unittest.main()
