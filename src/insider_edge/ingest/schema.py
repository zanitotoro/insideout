"""The canonical event record: one qualifying insider open-market buy.

Plan §6 Phase 0, reconciled with the real 2025q1 schema in Phase 1. `filing_date`
is the ONLY actionable date — entering on `txn_date` is look-ahead and is
forbidden (see analysis/returns.first_tradable_day, tests/test_no_lookahead.py).

Role is a *set* of flags, not one label: RPTOWNER_RELATIONSHIP is comma-joined
(an insider can be Director AND Officer). `price_footnoted` marks rows whose
reported price has a footnote ref and is therefore unreliable for size analysis.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class InsiderBuy(BaseModel):
    accession: str = Field(description="SEC ACCESSION_NUMBER — the filing key.")
    txn_sk: int = Field(description="NONDERIV_TRANS_SK — transaction key within the filing.")

    issuer_cik: str = Field(description="Issuer CIK (the permanent id to price-match on).")
    issuer_name: str | None = None
    ticker: str | None = Field(default=None, description="May be a placeholder for delisted names.")
    document_type: str = "4"

    insider_id: str = Field(description="RPTOWNERCIK — representative filer when co-filed.")
    insider_name: str | None = None
    relationship: str | None = Field(default=None, description="Raw RPTOWNER_RELATIONSHIP string.")
    title: str | None = None
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    is_other: bool = False
    n_reporting_owners: int = 1

    txn_date: date | None = Field(default=None, description="Trade date. DO NOT trade on this.")
    filing_date: date = Field(description="EDGAR filing date — the only actionable date.")
    period_of_report: date | None = None

    txn_code: str = Field(default="P", description="Kept == 'P' (open-market purchase).")
    shares: float
    price: float | None = None
    price_footnoted: bool = Field(
        default=False, description="Price has a footnote ref — unreliable."
    )
    value: float | None = Field(
        default=None, description="shares * price (unreliable if footnoted)."
    )
    shares_owned_following: float | None = None
    direct_indirect: str | None = None
    is_10b5_1: bool = Field(default=False, description="Scheduled plan trade — excluded upstream.")
