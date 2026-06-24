"""Async download + extraction of the quarterly SEC Form 345 data sets (plan §1a, §4.1).

The bulk sets are the backtest universe: they include filings from *delisted*
companies (it is all filings as-filed), which is essential for survivorship-free
work. A handful of files per year → modest concurrency, but the fair-access rules
(§4.2) still apply: < 10 req/s and a descriptive User-Agent with a contact email.
"""

from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter

from insider_edge import config

HEADERS = {"User-Agent": config.USER_AGENT}
_limiter = AsyncLimiter(max_rate=config.SEC_MAX_REQUESTS_PER_SEC, time_period=1.0)
_sem = asyncio.Semaphore(config.SEC_MAX_REQUESTS_PER_SEC)

BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"


def quarter_url(year: int, q: int) -> str:
    return f"{BASE}/{year}q{q}_form345.zip"


def quarters_range(
    start_year: int, start_q: int, end_year: int, end_q: int
) -> list[tuple[int, int]]:
    """Inclusive list of (year, quarter) from start to end."""
    out: list[tuple[int, int]] = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        out.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


async def _get(client: httpx.AsyncClient, url: str, attempts: int = 5) -> bytes:
    for i in range(attempts):
        async with _limiter, _sem:
            r = await client.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code in (403, 429, 500, 502, 503, 504):
            await asyncio.sleep(2**i + 0.1 * i)  # exponential backoff + jitter
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed after {attempts} attempts: {url}")


async def download_quarters(
    quarters: list[tuple[int, int]], outdir: Path | str | None = None
) -> list[Path]:
    """Download the given quarters' zips into `outdir` (default config.RAW_DIR)."""
    out = Path(outdir) if outdir is not None else config.RAW_DIR
    out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(http2=True) as c:
        blobs = await asyncio.gather(*(_get(c, quarter_url(y, q)) for y, q in quarters))
    paths: list[Path] = []
    for (y, q), b in zip(quarters, blobs, strict=True):
        p = out / f"{y}q{q}.zip"
        p.write_bytes(b)
        paths.append(p)
    return paths


def extract_zip(zip_path: Path | str, outdir: Path | str | None = None) -> Path:
    """Extract one quarterly zip into its own subdir so the 8 TSVs don't collide.

    Returns the directory the TSVs were written to (e.g. .../raw/2025q1/).
    """
    zip_path = Path(zip_path)
    out = Path(outdir) if outdir is not None else config.RAW_DIR
    dest = out / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest
