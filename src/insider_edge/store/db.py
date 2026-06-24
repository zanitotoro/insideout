"""DuckDB / parquet helpers (plan §5 store/db.py).

In-memory by default — the full Form 4 history plus daily prices fits in 32 GB
unified memory (plan §2). On a 16 GB unit, pass a file path to go out-of-core.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from insider_edge import config


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with threads pinned to the machine's core count."""
    con = duckdb.connect(database)
    con.execute(f"SET threads TO {config.N_THREADS}")
    return con


def write_parquet(df: pl.DataFrame, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path | str) -> pl.DataFrame:
    return pl.read_parquet(path)
