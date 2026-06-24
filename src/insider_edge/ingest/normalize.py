"""Normalize raw Form 4 buy rows into the canonical event table (plan §1b).

Responsibilities:
  * parse DD-MON-YYYY date strings into real dates,
  * normalize the messy AFF10B5ONE string and (optionally) scan footnotes for
    pre-2023 10b5-1 plans,
  * derive boolean role flags from the multi-role relationship string,
  * compute trade value,
  * collapse the co-filer fan-out to exactly one row per economic transaction
    (accession + NONDERIV_TRANS_SK), OR-ing role flags across co-filers.
"""

from __future__ import annotations

import polars as pl

from insider_edge.ingest.filters import add_role_flags, parse_10b5_1_flag

# SEC flat files use e.g. "31-MAR-2025". chrono's %b is case-insensitive.
_DATE_FMT = "%d-%b-%Y"


def _to_date(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.to_date(_DATE_FMT, strict=False)


def normalize_buys(df: pl.DataFrame, footnotes: pl.DataFrame | None = None) -> pl.DataFrame:
    """Turn raw `parse_buys` output into deduped, typed canonical events."""
    df = parse_10b5_1_flag(df)
    df = add_role_flags(df)
    df = df.with_columns(
        _to_date("filing_date_raw").alias("filing_date"),
        _to_date("txn_date_raw").alias("txn_date"),
        _to_date("period_raw").alias("period_of_report"),
        (pl.col("shares") * pl.col("price")).alias("value"),
        # A footnote ref on the price means the numeric field is often a placeholder
        # or weighted-average (sometimes junk, e.g. 1e6). Flag it; do not trust
        # `value` for size analysis where this is true.
        pl.col("price_fn").is_not_null().alias("price_footnoted"),
    )

    if footnotes is not None:
        df = _apply_footnote_10b5_1(df, footnotes)

    return _dedupe_transactions(df)


def _apply_footnote_10b5_1(df: pl.DataFrame, footnotes: pl.DataFrame) -> pl.DataFrame:
    """OR a footnote-text "10b5-1" match into is_10b5_1 (pre-2023 fallback)."""
    fn = (
        footnotes.group_by("ACCESSION_NUMBER")
        .agg(pl.col("FOOTNOTE_TXT").cast(pl.Utf8).str.join(" ").alias("_fn"))
        .with_columns(pl.col("_fn").str.contains(r"(?i)10b5-1").fill_null(False).alias("_fn_flag"))
        .select(pl.col("ACCESSION_NUMBER").alias("accession"), "_fn_flag")
    )
    return (
        df.join(fn, on="accession", how="left")
        .with_columns(
            (pl.col("is_10b5_1") | pl.col("_fn_flag").fill_null(False)).alias("is_10b5_1")
        )
        .drop("_fn_flag")
    )


def _dedupe_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (accession, txn_sk); OR role flags across co-filers.

    Joining on ACCESSION_NUMBER multiplies each transaction by the number of
    reporting owners. The transaction fields (issuer, dates, shares, price) are
    identical across those duplicate rows, so `first` is exact; role flags are
    unioned with `any`; a representative insider is taken (lowest CIK, stable).
    """
    df = df.sort(["accession", "txn_sk", "insider_id"])
    return df.group_by(["accession", "txn_sk"], maintain_order=True).agg(
        pl.col("issuer_cik").first(),
        pl.col("issuer_name").first(),
        pl.col("ticker").first(),
        pl.col("document_type").first(),
        pl.col("filing_date").first(),
        pl.col("txn_date").first(),
        pl.col("period_of_report").first(),
        pl.col("txn_code").first(),
        pl.col("shares").first(),
        pl.col("price").first(),
        pl.col("price_footnoted").first(),
        pl.col("value").first(),
        pl.col("shares_owned_following").first(),
        pl.col("direct_indirect").first(),
        pl.col("is_10b5_1").first(),
        pl.col("insider_id").first(),
        pl.col("insider_name").first(),
        pl.col("relationship").first(),
        pl.col("title").drop_nulls().first().alias("title"),
        pl.col("is_director").any(),
        pl.col("is_officer").any(),
        pl.col("is_ten_percent_owner").any(),
        pl.col("is_other").any(),
        pl.len().alias("n_reporting_owners"),
    )
