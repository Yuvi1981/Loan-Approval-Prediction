"""
Unit tests for data drift module.
Validates PSI calculation on identical and shifted distributions, and KS-test.
"""

import numpy as np
import pytest

from src.drift import calculate_ks_test, calculate_psi


def test_psi_identical_data_is_zero():
    data = np.random.normal(loc=100, scale=15, size=500)
    psi_identical = calculate_psi(data, data, bins=10)
    assert psi_identical == pytest.approx(0.0, abs=1e-3)


def test_psi_shifted_data_is_large():
    baseline = np.random.normal(loc=100, scale=15, size=500)
    shifted = np.random.normal(loc=160, scale=30, size=500)  # Significant shift

    psi_shifted = calculate_psi(baseline, shifted, bins=10)
    assert psi_shifted > 0.25, f"Expected significant drift (>0.25), got {psi_shifted}"


def test_psi_categorical():
    baseline = ["Urban"] * 50 + ["Rural"] * 30 + ["Semiurban"] * 20
    # Same distribution
    psi_same = calculate_psi(baseline, baseline, is_categorical=True)
    assert psi_same == pytest.approx(0.0, abs=1e-3)

    # Shifted distribution
    shifted = ["Urban"] * 10 + ["Rural"] * 80 + ["Semiurban"] * 10
    psi_diff = calculate_psi(baseline, shifted, is_categorical=True)
    assert psi_diff > 0.10


def test_ks_test_flags_drift():
    base = np.random.normal(0, 1, 300)
    shifted = np.random.normal(2, 1, 300)

    stat, pval, is_drift = calculate_ks_test(base, shifted)
    assert is_drift is True
    assert pval < 0.05
