"""Structural segment features (plan §6 Phase 3) — "is this *kind* of buy good?".

Segment, don't identify: each feature splits the population into buckets with
thousands of observations, so the answer is reliable and generalizable. No
per-individual selection here (that is Phase 5 shrinkage, and only past the
Phase 4 gate).

Features: role, trade size (vs insider's own median / vs company size), cluster
(distinct insiders buying the same issuer within a window), opportunistic vs.
routine (Cohen-Malloy-Pomorski), market-cap band, distance from 52-week high.
"""

from __future__ import annotations

import polars as pl


def add_segment_features(events: pl.DataFrame, *, cluster_window_days: int = 30) -> pl.DataFrame:
    """Attach segment columns to the event table."""
    raise NotImplementedError("Phase 3: role / size / cluster / routine / mcap features.")
