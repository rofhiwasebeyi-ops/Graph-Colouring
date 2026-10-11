"""
Small helper for summarising N stochastic runs (GA/ACO) per graph into
mean/std/min/max, and running a Mann-Whitney U test to check whether GA
and ACO's colour-count distributions differ significantly on a given graph.
Used by run_experiment.py and the report to back up claims with actual
statistics rather than eyeballing single numbers.
"""

import numpy as np
from scipy.stats import mannwhitneyu


def summarise(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def compare_methods(values_a, values_b):
    """
    Mann-Whitney U test (non-parametric, appropriate for small-sample,
    non-normal metaheuristic results) for whether two methods' results
    differ significantly. Returns (statistic, p_value).
    """
    values_a, values_b = np.asarray(values_a), np.asarray(values_b)
    if np.all(values_a == values_a[0]) and np.all(values_b == values_b[0]):
        # Both constant: if the constants are equal there is genuinely no
        # difference to detect (p=1.0). If they differ, the two methods
        # produced completely non-overlapping constant results -- that IS
        # a difference, just one the rank-based test can't score normally,
        # so report it as maximally significant rather than "no difference".
        if values_a[0] == values_b[0]:
            return None, 1.0
        return None, 0.0
    stat, p = mannwhitneyu(values_a, values_b, alternative="two-sided")
    return float(stat), float(p)
