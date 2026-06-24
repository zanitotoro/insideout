"""Shared fixtures for the bias-gate tests."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from insider_edge.analysis.returns import first_tradable_day


def _weekday_calendar(start: dt.date, end: dt.date) -> list[dt.date]:
    """A synthetic trading calendar: weekdays only (good enough for entry-timing tests)."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


@pytest.fixture
def trading_calendar() -> list[dt.date]:
    return _weekday_calendar(dt.date(2020, 1, 1), dt.date(2021, 1, 1))


@pytest.fixture
def trades(trading_calendar: list[dt.date]) -> pl.DataFrame:
    """Trades whose entry_date is derived only from filing_date — never txn_date."""
    filings = [
        dt.date(2020, 1, 1),
        dt.date(2020, 3, 14),  # Saturday -> entry rolls forward
        dt.date(2020, 7, 4),  # Saturday -> entry rolls forward
        dt.date(2020, 6, 15),
    ]
    rows = [
        {"filing_date": f, "entry_date": first_tradable_day(f, trading_calendar)} for f in filings
    ]
    return pl.DataFrame(rows)
