"""Alignment metrics between discovered splits and known ground truth."""

from __future__ import annotations

import numpy as np


def _entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def normalized_information_gain(true_D: np.ndarray, members: np.ndarray, group1: np.ndarray) -> float:
    t = true_D[members].astype(bool)
    h = _entropy(t.mean())
    if h == 0.0:
        return 0.0
    cond = 0.0
    for side in (group1, ~group1):
        if side.sum() == 0:
            continue
        cond += side.mean() * _entropy(t[side].mean())
    return max((h - cond) / h, 0.0)
