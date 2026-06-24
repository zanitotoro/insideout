"""Population & segment horizon curves with bootstrap CIs (plan §6 Phase 2).

Phase 2 is the FIRST decision gate: pool all buys and report mean abnormal return
per holding horizon with bootstrap confidence intervals. No positive, meaningful
drift before costs -> STOP. If it drifts, the curve's *shape* sets the holding
period.

Heavy resampling runs on the M5 GPU via MLX (arrays live in unified memory, so the
B x N index matrix is drawn and gathered with no host/device copy). Falls back to
NumPy where MLX is unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from insider_edge.analysis.returns import HORIZON_TRADING_DAYS

try:
    import mlx.core as mx
except ImportError:  # MLX needs Apple Silicon + macOS >= 13.3
    mx = None


def bootstrap_mean_ci(
    abn_returns: Sequence[float],
    n_boot: int = 10_000,
    lo: float = 2.5,
    hi: float = 97.5,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Point estimate of the mean and a percentile bootstrap CI.

    Returns (mean, (lo_pct, hi_pct)). Uses MLX on the GPU when available.
    """
    if len(abn_returns) == 0:
        raise ValueError("abn_returns is empty")
    if mx is not None:
        return _bootstrap_mlx(abn_returns, n_boot, lo, hi, seed)
    return _bootstrap_numpy(abn_returns, n_boot, lo, hi, seed)


def _bootstrap_mlx(abn_returns, n_boot, lo, hi, seed):
    mx.random.seed(seed)
    x = mx.array(list(abn_returns))  # lives in unified memory
    n = x.shape[0]
    idx = mx.random.randint(0, n, shape=(n_boot, n))  # B x N drawn on GPU
    means = mx.sort(mx.take(x, idx).mean(axis=1))
    mx.eval(means)  # force materialization
    return float(x.mean()), (
        float(means[int(lo / 100 * n_boot)]),
        float(means[int(hi / 100 * n_boot)]),
    )


def _bootstrap_numpy(abn_returns, n_boot, lo, hi, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(abn_returns, dtype=float)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.sort(x[idx].mean(axis=1))
    return float(x.mean()), (
        float(means[int(lo / 100 * n_boot)]),
        float(means[int(hi / 100 * n_boot)]),
    )


def population_curve(
    returns: pl.DataFrame, *, horizons: Sequence[str] | None = None, n_boot: int = 10_000
) -> pl.DataFrame:
    """Mean abnormal return per horizon over the whole population, with bootstrap CIs.

    `returns` is the output of analysis.returns.compute_abnormal_returns. Each row of
    the result is one horizon: n observations, mean abnormal return, and a 95%
    percentile-bootstrap CI. This is the Phase 2 gate input: if no horizon shows a
    positive CI that clears zero, there is no drift to trade (STOP).
    """
    horizons = list(horizons or HORIZON_TRADING_DAYS.keys())
    rows = []
    for h in horizons:
        col = f"abn_{h}"
        vals = returns.filter(pl.col(col).is_not_null())[col].to_list()
        if not vals:
            rows.append({"horizon": h, "n": 0, "mean_abn": None, "ci_lo": None, "ci_hi": None})
            continue
        mean, (lo, hi) = bootstrap_mean_ci(vals, n_boot=n_boot)
        rows.append({"horizon": h, "n": len(vals), "mean_abn": mean, "ci_lo": lo, "ci_hi": hi})
    return pl.DataFrame(rows)
