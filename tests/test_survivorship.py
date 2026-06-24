"""The survivorship gate (plan §1, §4.5): delisted names are booked as a real loss."""

from __future__ import annotations

import pytest

from insider_edge.analysis.returns import DELISTED_RETURN, forward_return


def test_delisting_is_total_loss_by_default():
    assert forward_return(10.0, 0.0, delisted=True) == DELISTED_RETURN == -1.0


def test_delisting_uses_documented_recovery_value():
    # entry 10, recovery 2 -> -80%, not silently dropped or forward-filled.
    assert forward_return(10.0, 999.0, delisted=True, recovery_value=2.0) == pytest.approx(-0.8)


def test_ordinary_return():
    assert forward_return(10.0, 12.0) == pytest.approx(0.2)


def test_dropping_delisted_inflates_the_mean():
    """Demonstrates the bias direction: silently deleting the failures looks great."""
    survivors = [0.10, 0.20, -0.05]
    naive_mean = sum(survivors) / len(survivors)  # delisted name deleted
    corrected = survivors + [DELISTED_RETURN]  # delisted booked at -100%
    corrected_mean = sum(corrected) / len(corrected)
    assert corrected_mean < naive_mean
