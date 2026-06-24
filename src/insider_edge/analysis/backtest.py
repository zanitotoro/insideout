"""Dumb-basket backtest with real costs, out-of-sample (plan §6 Phase 4) — GO/NO-GO.

Equal-weight basket of the signal (or best segments), fixed hold = the Phase-2
horizon, monthly rebalance. Frictions are MANDATORY: fills at next open (never
midpoint), bid-ask spread by liquidity bucket, market impact scaled to order size
vs ADV, commissions. Validate with time-split OOS / walk-forward; pre-register
rules; never tune on the test set. No edge net of costs OOS -> STOP.

Walk-forward folds are embarrassingly parallel -> one fold per core.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from insider_edge import config


def backtest_one_fold(fold):
    """Run a single walk-forward fold (defined in Phase 4)."""
    raise NotImplementedError("Phase 4: basket construction, costs, metrics.")


def run_walk_forward(folds: list) -> list:
    """Run independent folds across all cores."""
    with ProcessPoolExecutor(max_workers=config.N_THREADS) as ex:
        return list(ex.map(backtest_one_fold, folds))
