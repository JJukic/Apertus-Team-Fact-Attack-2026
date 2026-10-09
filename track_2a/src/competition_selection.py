"""Transparent component selection and *simulated* competition utilities.

Weights are supplied by the optimization objective. Competitor efficiency
references and task-specific official processing times are not available.
These functions must never be labeled an official final competition score.
"""

import statistics


def distribution(values):
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None}
    ordered = sorted(values)
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': ordered[round(0.95 * (len(ordered) - 1))]}


def quality_component(result):
    tasks = result['tasks']
    return (0.30 * tasks['A']['macro_f1'] + 0.12 * tasks['A']['evidence_hit5']
            + 0.28 * tasks['B']['macro_f1'])


def simulated_score(result, baseline, reference_fraction, own_time_factors=(1.0, 1.0)):
    """Hypothetical reference efficiencies relative to B0; token lower bounds.

    own_time_factors are explicit assumptions for A/B, never inferred from
    concurrent mixed-task wall time. reference_fraction models a competing
    team's cost as a fraction of B0. The best qualifying team includes ourselves,
    so efficiency cannot exceed 1 even if a hypothetical competitor is worse.
    """
    if not 0 < reference_fraction <= 1 or any(value <= 0 for value in own_time_factors):
        raise ValueError('Positive relative costs and a reference fraction in (0,1] are required')
    score = quality_component(result)
    for task, token_weight, time_weight, time_factor in (
            ('A', 0.09, 0.09, own_time_factors[0]), ('B', 0.06, 0.06, own_time_factors[1])):
        own = sum(result['tasks'][task][field] for field in ('input_tokens_reported', 'output_tokens_reported'))
        old = sum(baseline['tasks'][task][field] for field in ('input_tokens_reported', 'output_tokens_reported'))
        if own <= 0 or old <= 0:
            raise ValueError('Token scenarios require nonzero measured known usage')
        score += token_weight * min(1.0, reference_fraction * old / own)
        score += time_weight * min(1.0, reference_fraction / time_factor)
    return score


def known_component_pareto(summaries):
    """Frontier over known quality/token components only; timing is excluded."""
    def axes(result):
        return [result['tasks']['A']['macro_f1'], result['tasks']['B']['macro_f1'],
                result['tasks']['A']['evidence_hit5'],
                -sum(task[field] for task in result['tasks'].values()
                     for field in ('input_tokens_reported', 'output_tokens_reported'))]
    eligible = {name: axes(result) for name, result in summaries.items()
                if result.get('selection_eligible', result['qualifies'])}
    return [name for name, values in eligible.items()
            if not any(all(a >= b for a, b in zip(other, values)) and
                       any(a > b for a, b in zip(other, values))
                       for other_name, other in eligible.items() if other_name != name)]


def rank_by_known_quality(summaries):
    return sorted((name for name, result in summaries.items()
                   if result.get('selection_eligible', result['qualifies'])),
                  key=lambda name: quality_component(summaries[name]), reverse=True)
