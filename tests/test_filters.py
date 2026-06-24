"""Signal-definition filters (plan §4.3, §4.4)."""

from __future__ import annotations

import polars as pl

from insider_edge.ingest.filters import (
    add_role_flags,
    exclude_10b5_1,
    keep_open_market_purchases,
    parse_10b5_1_flag,
)


def test_keep_only_open_market_purchases():
    df = pl.DataFrame(
        {
            "TRANS_CODE": ["P", "S", "A", "P", "M", "P"],
            "TRANS_ACQUIRED_DISP_CD": ["A", "D", "A", "A", "A", "D"],
        }
    )
    out = keep_open_market_purchases(df)
    # Only rows that are BOTH code 'P' AND acquired 'A' survive (rows 0 and 3).
    assert out.height == 2
    assert set(out["TRANS_CODE"].to_list()) == {"P"}
    assert set(out["TRANS_ACQUIRED_DISP_CD"].to_list()) == {"A"}


def test_exclude_10b5_1_via_flag():
    df = pl.DataFrame({"TRANS_CODE": ["P", "P"], "is_10b5_1": [True, False]})
    out = exclude_10b5_1(df)
    assert out.height == 1
    assert out["is_10b5_1"].to_list() == [False]


def test_exclude_10b5_1_via_footnote_scan():
    df = pl.DataFrame(
        {
            "TRANS_CODE": ["P", "P", "P"],
            "footnote": ["Made pursuant to a Rule 10b5-1 plan", "routine open-market buy", None],
        }
    )
    out = exclude_10b5_1(df, footnote_col="footnote")
    assert out.height == 2  # the 10b5-1 row is dropped; the None footnote is kept


def test_parse_10b5_1_flag_handles_messy_strings():
    # AFF10B5ONE comes in as '0' / 'false' / '1' / 'true' / null across filings.
    df = pl.DataFrame({"aff10b5one": ["1", "true", "TRUE", "0", "false", None]})
    out = parse_10b5_1_flag(df)
    assert out["is_10b5_1"].to_list() == [True, True, True, False, False, False]


def test_parse_10b5_1_flag_missing_column_defaults_false():
    df = pl.DataFrame({"x": [1, 2]})  # pre-2023 quarters lack the column entirely
    out = parse_10b5_1_flag(df)
    assert out["is_10b5_1"].to_list() == [False, False]


def test_add_role_flags_parses_multi_role_string():
    df = pl.DataFrame({"relationship": ["Director,Officer", "TenPercentOwner", "Other", None]})
    out = add_role_flags(df)
    assert out["is_director"].to_list() == [True, False, False, False]
    assert out["is_officer"].to_list() == [True, False, False, False]
    assert out["is_ten_percent_owner"].to_list() == [False, True, False, False]
    assert out["is_other"].to_list() == [False, False, True, False]
