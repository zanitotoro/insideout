"""No-lookahead entry timing and survivorship-correct forward returns.

The two cheapest ways to fabricate an edge:
  1. Look-ahead — entering on the (earlier) transaction date instead of the public
     filing date. `first_tradable_day` makes entry the first session on/after the
     filing date, so entry_date >= filing_date by construction.
  2. Survivorship — dropping delisted names silently deletes the buys that went to
     zero. `forward_return` books a delisting as -100% (or a documented recovery
     value), never a gap or a forward-filled price.

Both gates are asserted in tests/test_no_lookahead.py and tests/test_survivorship.py.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from datetime import date

import polars as pl

# A delisting is total loss unless a recovery value is documented (plan §4.5).
DELISTED_RETURN = -1.0

# Approximate trading-day counts per horizon (plan §2/§6). The *shape* of the
# population horizon curve, not these labels, ultimately sets the holding period.
HORIZON_TRADING_DAYS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "12m": 252}


def first_tradable_day(filing_date: date, trading_calendar: Sequence[date]) -> date | None:
    """First trading day on or after `filing_date` (the only legal entry day).

    You may only act on public information — the filing date — never the earlier
    transaction date. Entry is the next session's open: the first trading day
    >= filing_date. `trading_calendar` must be sorted ascending. Returns None if
    the calendar ends before the filing date.
    """
    i = bisect_left(trading_calendar, filing_date)
    if i >= len(trading_calendar):
        return None
    return trading_calendar[i]


def forward_return(
    entry_price: float,
    exit_price: float,
    *,
    delisted: bool = False,
    recovery_value: float = 0.0,
) -> float:
    """Total return from entry to exit, treating a delisting as a real loss.

    A delisted name is not a missing point to drop or forward-fill; it is realized
    at `recovery_value` (default 0.0 -> -100%).
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if delisted:
        return recovery_value / entry_price - 1.0
    return exit_price / entry_price - 1.0


def _calendar_frame(calendar: Sequence[date]) -> pl.DataFrame:
    """Trading calendar as (t, date) with an integer session index for horizon math."""
    dates = sorted(calendar)
    return pl.DataFrame({"t": list(range(len(dates))), "cal_date": dates}).with_columns(
        pl.col("cal_date").cast(pl.Date)
    )


def compute_abnormal_returns(
    events: pl.DataFrame,
    prices: pl.DataFrame,
    benchmark: pl.DataFrame,
    calendar: Sequence[date],
    horizons: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Per-event abnormal returns at each horizon, indexed off the FILING date.

    No look-ahead: entry is the first trading session on/after `filing_date`
    (`join_asof` forward); exit is `entry + horizon` sessions. Abnormal return =
    stock total return − benchmark total return over the same window. `prices` and
    `benchmark` are long frames [ticker?, date, close]; returns the events frame
    with an `entry_date`, plus `abn_<h>` / `ret_<h>` columns and a `priced` flag.

    Events whose ticker is missing from `prices` (e.g. delisted names absent from a
    survivorship-biased source) come back with null returns and priced=False — the
    bias is surfaced, never hidden.
    """
    horizons = horizons or HORIZON_TRADING_DAYS
    cal = _calendar_frame(calendar)
    max_t = cal["t"].max()

    px = prices.select("ticker", pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64))
    bench = benchmark.select(
        pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64).alias("bclose")
    )

    # Entry session = first calendar date >= filing_date (forward asof join).
    ev = (
        events.select("accession", "txn_sk", "ticker", pl.col("filing_date").cast(pl.Date))
        .sort("filing_date")
        .join_asof(
            cal.sort("cal_date"), left_on="filing_date", right_on="cal_date", strategy="forward"
        )
        .rename({"t": "entry_t", "cal_date": "entry_date"})
    )

    out = ev
    for label, h in horizons.items():
        # exit session index = entry_t + h; map (entry_t) -> the date h sessions later
        ladder = cal.select(
            (pl.col("t") - h).alias("entry_t"), pl.col("cal_date").alias(f"exit_date_{label}")
        )
        out = out.join(ladder, on="entry_t", how="left")

    # Attach entry/exit prices for stock and benchmark, then compute returns.
    out = out.join(
        px.rename({"date": "entry_date", "close": "entry_close"}),
        on=["ticker", "entry_date"],
        how="left",
    )
    out = out.join(
        bench.rename({"date": "entry_date", "bclose": "bench_entry"}), on="entry_date", how="left"
    )

    ret_exprs = []
    for label in horizons:
        out = out.join(
            px.rename({"date": f"exit_date_{label}", "close": f"exit_close_{label}"}),
            on=["ticker", f"exit_date_{label}"],
            how="left",
        ).join(
            bench.rename({"date": f"exit_date_{label}", "bclose": f"bench_exit_{label}"}),
            on=f"exit_date_{label}",
            how="left",
        )
        stock_ret = pl.col(f"exit_close_{label}") / pl.col("entry_close") - 1.0
        bench_ret = pl.col(f"bench_exit_{label}") / pl.col("bench_entry") - 1.0
        ret_exprs.append(stock_ret.alias(f"ret_{label}"))
        ret_exprs.append((stock_ret - bench_ret).alias(f"abn_{label}"))

    out = out.with_columns(ret_exprs)
    out = out.with_columns(
        (pl.col("entry_close").is_not_null() & (pl.col("entry_t") <= max_t)).alias("priced")
    )
    keep = ["accession", "txn_sk", "ticker", "filing_date", "entry_date", "entry_close", "priced"]
    keep += [f"ret_{h}" for h in horizons] + [f"abn_{h}" for h in horizons]
    return out.select(keep)
