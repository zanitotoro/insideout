"""Keep only genuine open-market insider purchases (plan §4.3, §4.4).

These filters *define the signal*, so the bias gates depend on them and they are
unit-tested in tests/test_filters.py.
"""

from __future__ import annotations

import polars as pl

# §4.3 — the only transaction code that is a conviction open-market BUY.
KEEP_TRANS_CODE = "P"
# Acquired (vs. Disposed). A buy must be an acquisition.
ACQUIRED = "A"

# Everything else, with the reason, for documentation and sanity-count checks.
EXCLUDED_TRANS_CODES = {
    "S": "open-market/private sale (noisy: taxes, diversification, liquidity)",
    "A": "grant/award/acquisition from the company (compensation, not conviction)",
    "M": "exercise/conversion of derivative (16b-3 exempt)",
    "F": "shares withheld to pay exercise price / tax",
    "G": "bona fide gift",
    "X": "exercise of in-the-money derivative",
}


def keep_open_market_purchases(
    df: pl.DataFrame,
    *,
    code_col: str = "TRANS_CODE",
    ad_col: str = "TRANS_ACQUIRED_DISP_CD",
) -> pl.DataFrame:
    """Keep rows that are open-market purchases AND acquisitions (code 'P', A/D 'A')."""
    return df.filter((pl.col(code_col) == KEEP_TRANS_CODE) & (pl.col(ad_col) == ACQUIRED))


# AFF10B5ONE is a string with mixed encodings across filings; these mean "true".
TRUE_STRINGS = ["1", "true", "t", "yes", "y"]


def parse_10b5_1_flag(
    df: pl.DataFrame, *, col: str = "aff10b5one", out: str = "is_10b5_1"
) -> pl.DataFrame:
    """Normalize the messy AFF10B5ONE string to a clean boolean `out` column.

    Values seen in the wild: '0', 'false', '1', 'true', null. null (no affirmation
    / pre-2023 filings) becomes False here; pre-2023 detection is handled by the
    footnote scan in normalize.py.
    """
    if col not in df.columns:
        return df.with_columns(pl.lit(False).alias(out))
    flag = (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .is_in(TRUE_STRINGS)
        .fill_null(False)
        .alias(out)
    )
    return df.with_columns(flag)


def add_role_flags(df: pl.DataFrame, *, col: str = "relationship") -> pl.DataFrame:
    """Split the comma-joined RPTOWNER_RELATIONSHIP into boolean role flags.

    Segmentation (plan §6 Phase 3) asks "is this *kind* of buy good?", so role is a
    set of flags (an insider can be Director AND Officer), not a single label.
    """
    rel = pl.col(col).cast(pl.Utf8).fill_null("").str.to_lowercase()
    return df.with_columns(
        rel.str.contains("director").alias("is_director"),
        rel.str.contains("officer").alias("is_officer"),
        rel.str.contains("tenpercent").alias("is_ten_percent_owner"),
        rel.str.contains("other").alias("is_other"),
    )


def exclude_10b5_1(
    df: pl.DataFrame,
    *,
    flag_col: str = "is_10b5_1",
    footnote_col: str | None = None,
) -> pl.DataFrame:
    """Drop pre-planned Rule 10b5-1 trades (low signal — they are scheduled).

    Post-2023 filings carry an explicit affirmation flag (plan §4.4); when
    `flag_col` is present we drop where it is true. Pre-2023 there is no flag, so
    pass `footnote_col` to additionally drop rows whose footnote text mentions
    "10b5-1" (case-insensitive). Be conservative: when in doubt, drop.
    """
    if flag_col in df.columns:
        df = df.filter(~pl.col(flag_col).fill_null(False))
    if footnote_col is not None and footnote_col in df.columns:
        df = df.filter(~pl.col(footnote_col).fill_null("").str.contains(r"(?i)10b5-1"))
    return df
