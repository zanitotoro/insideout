"""PHASE 5 ONLY — per-insider records via hierarchical Bayesian shrinkage.

Do not run this before the Phase 4 gate passes: per-individual mining is the one
step that can only *invent* an edge through overfitting.

Partial pooling: start each insider at their segment mean; let the estimate
deviate only in proportion to trade count (4 trades -> pulled back to the group;
60 trades -> trusted). The real test is whether adding individual identity
improves OUT-OF-SAMPLE prediction over the segment model. Implement in MLX
(autodiff on GPU) or NumPyro/PyMC.
"""

from __future__ import annotations


def fit_shrinkage(events, segment_means):
    raise NotImplementedError("Phase 5 only — after the Phase 4 GO/NO-GO gate.")
