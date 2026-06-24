"""Swappable daily-price interface (plan §4.5) — the #1 risk in the project.

The verdict requires *survivorship-bias-free* prices that include delisted tickers,
plus corporate actions for adjusted returns. yfinance lacks delisted names and so
biases results upward — it is a SMOKE TEST only, never the verdict. Keep providers
behind `PriceProvider` so the layer can be upgraded (Sharadar / CRSP / Norgate /
Polygon) without touching the analysis code.

A delisting must be booked as -100% (or a documented recovery value), never as a
silent gap or a forward-filled last price — see analysis/returns.forward_return.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PriceProvider(Protocol):
    """Minimal contract every price source must satisfy."""

    def daily_closes(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        """Long frame [ticker, date, open, close] of split/div-adjusted prices."""
        ...

    def trading_calendar(self, start: date, end: date) -> list[date]:
        """Sorted trading days in [start, end] (used for no-lookahead entry)."""
        ...

    def delisting(self, ticker: str) -> tuple[date, float] | None:
        """(delist_date, recovery_value) if delisted, else None."""
        ...


class YFinanceProvider:
    """SMOKE TEST ONLY — lacks delisted names, so it silently inflates returns.

    Never use this provider to produce the final verdict (plan §4.5). It exists to
    exercise the Phase 2 machinery end-to-end on free data.
    """

    benchmark = "SPY"

    def daily_closes(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        import yfinance as yf

        clean = sorted({t for t in tickers if t and t.isascii() and t.replace(".", "").isalnum()})
        if not clean:
            return pl.DataFrame(
                schema={"ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64, "close": pl.Float64}
            )
        raw = yf.download(
            clean,
            start=start,
            end=end + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        return _yf_to_long(raw, clean)

    def trading_calendar(self, start: date, end: date) -> list[date]:
        df = self.daily_closes([self.benchmark], start, end)
        return df["date"].sort().to_list()

    def delisting(self, ticker: str) -> tuple[date, float] | None:
        return None  # yfinance cannot tell us — exactly why it is unfit for the verdict.


def _yf_to_long(raw, tickers: list[str]) -> pl.DataFrame:
    """Reshape a yfinance download (wide, multi-index columns) into a long frame."""
    import pandas as pd

    frames = []
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        if sub is None or sub.empty or "Close" not in sub:
            continue
        part = (
            sub[["Open", "Close"]]
            .reset_index()
            .rename(columns={"Date": "date", "Open": "open", "Close": "close"})
        )
        part["ticker"] = t
        frames.append(part.dropna(subset=["close"]))
    if not frames:
        return pl.DataFrame(
            schema={"ticker": pl.Utf8, "date": pl.Date, "open": pl.Float64, "close": pl.Float64}
        )
    out = pl.from_pandas(pd.concat(frames, ignore_index=True))
    return out.with_columns(pl.col("date").cast(pl.Date)).select("ticker", "date", "open", "close")
