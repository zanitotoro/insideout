"""The look-ahead gate (plan §1, §6 Phase 2): you may only act on the filing date."""

from __future__ import annotations

import datetime as dt

from insider_edge.analysis.returns import first_tradable_day


def test_entry_never_before_filing(trades):
    """The gate, as code: no entry may precede its filing date."""
    assert (trades["entry_date"] >= trades["filing_date"]).all()


def test_first_tradable_day_is_on_or_after(trading_calendar):
    for f in [dt.date(2020, 2, 29), dt.date(2020, 7, 4), dt.date(2020, 6, 15)]:
        e = first_tradable_day(f, trading_calendar)
        assert e is not None
        assert e >= f


def test_weekend_filing_rolls_forward(trading_calendar):
    # 2020-07-04 is a Saturday; entry must be the next weekday (Monday the 6th).
    assert first_tradable_day(dt.date(2020, 7, 4), trading_calendar) == dt.date(2020, 7, 6)


def test_past_calendar_end_returns_none(trading_calendar):
    assert first_tradable_day(dt.date(2099, 1, 1), trading_calendar) is None
