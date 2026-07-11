"""Subset-recovery metrics for the DiD benchmarks (Fig 6.1/6.2 analogs)."""

from __future__ import annotations

import numpy as np

_NAN = float("nan")


def subset_precision_recall(
    pred_mask: np.ndarray, true_mask: np.ndarray
) -> tuple[float, float, float]:
    """(precision, recall, F) of a predicted record mask against the truth.

    Undefined ratios are NaN, never 0/0 -> 0: an empty prediction has NaN
    precision, an empty truth has NaN recall, and F is NaN whenever either
    component is undefined. F is 0.0 only for the genuine zero-overlap limit
    (both masks nonempty, no intersection).
    """
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {true.shape}")
    tp = float(np.count_nonzero(pred & true))
    n_pred = float(np.count_nonzero(pred))
    n_true = float(np.count_nonzero(true))
    precision = tp / n_pred if n_pred > 0 else _NAN
    recall = tp / n_true if n_true > 0 else _NAN
    if np.isnan(precision) or np.isnan(recall):
        f = _NAN
    elif precision + recall == 0.0:
        f = 0.0
    else:
        f = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f
