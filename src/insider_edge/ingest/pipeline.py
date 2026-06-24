"""Phase 1 deliverable: raw TSVs -> qualifying events.parquet + sanity report.

A "qualifying" buy is an open-market purchase (code P, acquired) that is NOT a
pre-planned Rule 10b5-1 trade. The sanity report is the Phase 1 gate: eyeball the
buy/sell ratio, the by-year distribution, and a few rows against the live EDGAR
filing before trusting the data.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from insider_edge import config
from insider_edge.ingest.filters import exclude_10b5_1
from insider_edge.ingest.normalize import normalize_buys
from insider_edge.ingest.parse_form345 import code_distribution, load_footnotes, parse_buys
from insider_edge.store.db import write_parquet


def build_events(
    raw_root: str | None = None,
    out_path: Path | str | None = None,
    *,
    document_types: tuple[str, ...] = ("4",),
    scan_footnotes: bool = False,
) -> pl.DataFrame:
    """Parse, normalize, drop 10b5-1, and write events.parquet. Returns the events.

    `scan_footnotes=True` adds the pre-2023 footnote 10b5-1 fallback (slower; only
    needed for quarters where AFF10B5ONE is absent/null).
    """
    raw_root = raw_root if raw_root is not None else str(config.RAW_DIR)
    raw = parse_buys(raw_root, document_types=document_types)
    footnotes = load_footnotes(raw_root) if scan_footnotes else None
    events_all = normalize_buys(raw, footnotes)
    events = exclude_10b5_1(events_all)

    out_path = Path(out_path) if out_path is not None else config.PARQUET_DIR / "events.parquet"
    write_parquet(events, out_path)
    return events


def sanity_report(events: pl.DataFrame, raw_root: str | None = None) -> dict:
    """Compute the Phase 1 sanity numbers used to decide whether ingestion is trustworthy."""
    codes = code_distribution(raw_root)
    n_buys = int(codes.filter(pl.col("TRANS_CODE") == "P")["n"].sum())
    n_sells = int(codes.filter(pl.col("TRANS_CODE") == "S")["n"].sum())

    by_year = (
        events.with_columns(pl.col("filing_date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.len().alias("n"))
        .sort("year")
    )
    return {
        "n_events": events.height,
        "distinct_issuers": events["issuer_cik"].n_unique(),
        "distinct_insiders": events["insider_id"].n_unique(),
        "filing_date_min": events["filing_date"].min(),
        "filing_date_max": events["filing_date"].max(),
        "buy_sell_ratio": round(n_buys / n_sells, 3) if n_sells else None,
        "raw_purchases": n_buys,
        "raw_sales": n_sells,
        "null_or_zero_price": int(
            events.filter((pl.col("price").is_null()) | (pl.col("price") == 0)).height
        ),
        "footnoted_price": int(events["price_footnoted"].sum()),
        "suspect_price_gt_10k": int(events.filter(pl.col("price") > 10_000).height),
        "multi_owner_filings": int(events.filter(pl.col("n_reporting_owners") > 1).height),
        "role_counts": {
            "director": int(events["is_director"].sum()),
            "officer": int(events["is_officer"].sum()),
            "ten_percent": int(events["is_ten_percent_owner"].sum()),
            "other": int(events["is_other"].sum()),
        },
        "by_year": {
            int(r["year"]): int(r["n"]) for r in by_year.to_dicts() if r["year"] is not None
        },
    }


def edgar_filing_url(accession: str, cik: str | int) -> str:
    """Human-checkable EDGAR filing-index URL (hand-verify a few rows, plan §1b)."""
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{accession}-index.htm"
