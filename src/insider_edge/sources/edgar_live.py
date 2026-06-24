"""Recent-gap + ongoing Form 4 signal from EDGAR live (plan §4.2).

The quarterly bulk sets lag, so the window since the last published quarter is
fetched from EDGAR's daily index directly. Obey fair-access: < 10 req/s and the
descriptive User-Agent from config. Implemented in Phase 1b / as ongoing signal.

CIK gotcha: data.sec.gov endpoints need the CIK zero-padded to 10 digits
(320193 -> 'CIK0000320193'); un-padded returns 404.
"""

from __future__ import annotations

from datetime import date


def pad_cik(cik: int | str) -> str:
    """Zero-pad a CIK to the 10-digit form data.sec.gov requires."""
    return f"CIK{int(cik):010d}"


async def fetch_form4_since(last_quarter_end: date):
    """Fetch Form 4 filings from the daily index since the last published quarter."""
    raise NotImplementedError("Phase 1b: parse daily-index form.idx, filter Form Type == 4.")
