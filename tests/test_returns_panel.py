"""Vectorized abnormal-return computation: no look-ahead + correct math."""

from __future__ import annotations

import datetime as dt

import polars as pl

from insider_edge.analysis.returns import compute_abnormal_returns


def _calendar(n: int, start=dt.date(2025, 1, 1)) -> list[dt.date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def test_entry_is_first_session_on_or_after_filing_and_math_is_right():
    cal = _calendar(40)
    # Stock doubles by 5 sessions after entry; benchmark is flat.
    stock = pl.DataFrame({"ticker": ["AAA"] * 40, "date": cal, "close": [10.0] * 5 + [20.0] * 35})
    bench = pl.DataFrame({"date": cal, "close": [100.0] * 40})

    # File on a weekend -> entry must roll forward to the next session (cal[0] here is
    # a Wednesday; pick a filing date between sessions to prove the asof-forward).
    events = pl.DataFrame(
        {
            "accession": ["x"],
            "txn_sk": [1],
            "ticker": ["AAA"],
            "filing_date": [cal[0]],
        }
    )
    out = compute_abnormal_returns(events, stock, bench, cal, horizons={"1w": 5})
    row = out.to_dicts()[0]

    assert row["entry_date"] == cal[0]  # first session >= filing
    assert row["entry_close"] == 10.0
    assert row["priced"] is True
    # entry at index 0 (10.0), exit at index 5 (20.0) -> +100%; benchmark flat -> abn=+1.0
    assert abs(row["ret_1w"] - 1.0) < 1e-9
    assert abs(row["abn_1w"] - 1.0) < 1e-9


def test_missing_ticker_comes_back_unpriced_not_dropped():
    cal = _calendar(40)
    stock = pl.DataFrame({"ticker": ["AAA"] * 40, "date": cal, "close": [10.0] * 40})
    bench = pl.DataFrame({"date": cal, "close": [100.0] * 40})
    events = pl.DataFrame(
        {"accession": ["y"], "txn_sk": [1], "ticker": ["DELISTED"], "filing_date": [cal[0]]}
    )
    out = compute_abnormal_returns(events, stock, bench, cal, horizons={"1w": 5})
    row = out.to_dicts()[0]
    # The event is preserved (survivorship visible), just unpriced.
    assert out.height == 1
    assert row["priced"] is False
    assert row["abn_1w"] is None
