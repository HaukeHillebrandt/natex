"""Honest discovery/estimation splitting — the primary post-selection guarantee
(audit: cleaner than any claim of exactness for the fitted-null test)."""

from __future__ import annotations

import numpy as np


def honest_split(n: int, frac_discovery: float = 0.5, rng: np.random.Generator | None = None):
    if rng is None:
        raise ValueError("pass an explicit numpy Generator")
    perm = rng.permutation(n)
    cut = int(round(frac_discovery * n))
    return np.sort(perm[:cut]), np.sort(perm[cut:])
