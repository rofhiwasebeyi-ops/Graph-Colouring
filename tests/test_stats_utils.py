import numpy as np
from stats_utils import summarise, compare_methods


def test_summarise_basic():
    result = summarise([4, 4, 5, 5])
    assert result["mean"] == 4.5
    assert result["min"] == 4
    assert result["max"] == 5
    assert result["n"] == 4


def test_summarise_constant_values_zero_std():
    result = summarise([7, 7, 7])
    assert result["std"] == 0.0
    assert result["mean"] == 7.0


def test_compare_methods_identical_distributions_high_p():
    a = [5, 5, 5, 5, 5]
    b = [5, 5, 5, 5, 5]
    stat, p = compare_methods(a, b)
    assert p == 1.0  # both constant and equal -> defined as no difference


def test_compare_methods_clearly_different_distributions_low_p():
    a = [1, 1, 1, 1, 1, 1]
    b = [10, 10, 10, 10, 10, 10]
    stat, p = compare_methods(a, b)
    assert p < 0.05


def test_compare_methods_returns_valid_pvalue_range():
    a = [4, 5, 4, 6, 5]
    b = [5, 6, 5, 7, 6]
    stat, p = compare_methods(a, b)
    assert 0.0 <= p <= 1.0
