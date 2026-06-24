"""Phase 2 driver — population horizon curve (the FIRST decision gate, plan §6).

Glue only: load events -> fetch prices via the swappable provider -> compute
no-lookahead abnormal returns -> bootstrap the population curve. With the default
YFinanceProvider this is a SMOKE TEST (no delisted names); the coverage figure it
reports makes the survivorship hole explicit.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from insider_edge.analysis.horizon_curve import population_curve
from insider_edge.analysis.returns import compute_abnormal_returns
from insider_edge.sources.prices import PriceProvider, YFinanceProvider


def run_horizon(
    events: pl.DataFrame,
    provider: PriceProvider | None = None,
    *,
    sample: int | None = 400,
    seed: int = 1,
    n_boot: int = 10_000,
) -> dict:
    """Return {curve, coverage, n_events, n_priced, returns} for the event set."""
    provider = provider or YFinanceProvider()

    ev = events.filter(
        pl.col("ticker").is_not_null()
        & (pl.col("ticker").str.len_chars() > 0)
        & (pl.col("ticker").str.contains(r"^[A-Za-z.]+$"))
    )
    if sample is not None and ev.height > sample:
        ev = ev.sample(sample, seed=seed)

    tickers = ev["ticker"].unique().to_list()
    start = ev["filing_date"].min()
    # 12-month horizon needs ~1 calendar year of forward prices after the last filing.
    end = ev["filing_date"].max() + timedelta(days=420)

    prices = provider.daily_closes(tickers, start, end)
    bench_long = provider.daily_closes([getattr(provider, "benchmark", "SPY")], start, end)
    benchmark = bench_long.select("date", "close")
    calendar = provider.trading_calendar(start, end)

    returns = compute_abnormal_returns(ev, prices, benchmark, calendar)
    n_priced = int(returns.filter(pl.col("priced"))["priced"].len())
    curve = population_curve(returns, n_boot=n_boot)

    return {
        "curve": curve,
        "coverage": round(n_priced / ev.height, 3) if ev.height else None,
        "n_events": ev.height,
        "n_priced": n_priced,
        "returns": returns,
    }
