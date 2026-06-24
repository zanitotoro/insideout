"""Paths, SEC fair-access settings, and thread caps.

Centralised so the SEC contact email (legally required in the User-Agent) and the
data locations live in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- SEC fair-access (plan §4.2) ----------------------------------------------
# EDGAR REQUIRES a descriptive User-Agent containing a real contact email. Without
# it requests get HTTP 403 and the IP can be temporarily blocked. Override the
# contact via the env var if someone else runs this.
CONTACT_EMAIL = os.environ.get("INSIDER_EDGE_CONTACT", "stefano.zaninetta@proton.me")
USER_AGENT = f"insider-edge research {CONTACT_EMAIL}"

# Fair-access cap is 10 requests/second per IP; stay comfortably under it.
SEC_MAX_REQUESTS_PER_SEC = 8

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("INSIDER_EDGE_DATA", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # extracted quarterly Form 345 TSVs
PARQUET_DIR = DATA_DIR / "parquet"  # flattened event store

# --- Compute (plan §2) --------------------------------------------------------
# Match the machine's core count (M5 base = 10). Used for DuckDB `SET threads` and
# ProcessPoolExecutor sizing. polars auto-detects, but honour an explicit override.
N_THREADS = int(os.environ.get("INSIDER_EDGE_THREADS", os.cpu_count() or 8))


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    for d in (DATA_DIR, RAW_DIR, PARQUET_DIR):
        d.mkdir(parents=True, exist_ok=True)
