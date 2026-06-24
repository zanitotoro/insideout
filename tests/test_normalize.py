"""Normalize/dedup gate (plan §4.1): co-filer fan-out must not double-count buys."""

from __future__ import annotations

import datetime as dt

import polars as pl

from insider_edge.ingest.normalize import normalize_buys


def _raw_row(**kw):
    """A raw parse_buys-shaped row with sensible defaults; override via kwargs."""
    base = {
        "accession": "0001-25-000001",
        "filing_date_raw": "31-MAR-2025",
        "period_raw": "27-MAR-2025",
        "document_type": "4",
        "issuer_cik": "320193",
        "issuer_name": "Example Inc.",
        "ticker": "EXM",
        "aff10b5one": "0",
        "insider_id": "111",
        "insider_name": "Alice",
        "relationship": "Officer",
        "title": "CFO",
        "txn_sk": 1,
        "txn_date_raw": "27-MAR-2025",
        "txn_code": "P",
        "shares": 100.0,
        "price": 10.0,
        "price_fn": None,
        "ad": "A",
        "shares_owned_following": 1000.0,
        "direct_indirect": "D",
    }
    base.update(kw)
    return base


def test_dates_parse_uppercase_month():
    out = normalize_buys(pl.DataFrame([_raw_row()]))
    assert out["filing_date"][0] == dt.date(2025, 3, 31)
    assert out["txn_date"][0] == dt.date(2025, 3, 27)


def test_value_is_shares_times_price():
    out = normalize_buys(pl.DataFrame([_raw_row(shares=100.0, price=10.0)]))
    assert out["value"][0] == 1000.0


def test_price_footnote_flagged():
    out = normalize_buys(pl.DataFrame([_raw_row(price_fn="F1")]))
    assert out["price_footnoted"][0] is True


def test_cofiler_fanout_collapses_to_one_row_with_unioned_roles():
    # Same transaction (accession+txn_sk) reported by two co-filers with different
    # roles. Must collapse to ONE row, with the roles OR-ed together.
    rows = [
        _raw_row(insider_id="111", relationship="Officer"),
        _raw_row(insider_id="222", relationship="Director,TenPercentOwner"),
    ]
    out = normalize_buys(pl.DataFrame(rows))
    assert out.height == 1
    r = out.to_dicts()[0]
    assert r["n_reporting_owners"] == 2
    assert r["is_officer"] and r["is_director"] and r["is_ten_percent_owner"]
    assert r["shares"] == 100.0  # not doubled


def test_distinct_transactions_are_kept_separate():
    rows = [_raw_row(txn_sk=1, price=10.0), _raw_row(txn_sk=2, price=11.0)]
    out = normalize_buys(pl.DataFrame(rows))
    assert out.height == 2
