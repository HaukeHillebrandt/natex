"""Backtest: Egger & Koethenbuerger (2010) — Bavarian council-size thresholds (stretch).

Spec section 8 row 5 / section 10 stretch goal: LoRD3 was not designed for many
parallel cutoffs; the target is >= 2 of the statutory population thresholds
{1001, 2001, 3001, 5001, 10001} (council size ``rcsize`` jumps 8 -> 12 -> 14 ->
16 -> 20 -> 24 there), discovered on forcing ``log_pop`` with the normal model
(integer-valued treatment).

Test protocol (documented restriction): rows are restricted to ``wpop < 20000``
BEFORE scanning — the municipality-size distribution is heavily right-tailed
(max wpop ~ 1.28M), and without the cap the low statutory thresholds are
drowned by the tail. The restriction keeps 42,005 of 43,175 rows and is part of
the protocol, not tuning-to-pass: it was fixed in the plan before the scan ran.

OUTCOME (for docs/status/phase-2.md, task 12): the stretch goal PASSES STRICT
with zero tuning — no xfail needed. With k=100, n_coarse=3000, top_m=30,
seed 0, 4 of 5 thresholds ({2001, 3001, 5001, 10001}) have a top-15 cluster
representative within 15%; in fact they are the ONLY four clusters formed by
the top-50 discoveries (top cluster pops 4994 / 9992 / 1999 / 2995, max LLRs
~70456 / 6085 / 5716 / 138). The fifth threshold (1001) is not lost: its best
discovery sits at overall rank 54 (pop 1000, LLR ~116), i.e. cluster #5 by LLR
when clustering ALL fine-stage discoveries — all five statutory thresholds are
the top five clusters of the full discovery list; 1001 merely falls just
outside the plan's top-50 discovery cut. Full coarse-to-fine scan wall-clock:
~4 s (42k rows, 1-D forcing, normal model).
"""

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.rdd.metrics import cluster_discoveries
from natex.scan.coarse import coarse_to_fine_scan

pytestmark = pytest.mark.backtest

THRESHOLDS = (1001, 2001, 3001, 5001, 10001)
POP_CAP = 20000
REL_TOL = 0.15


@pytest.fixture(scope="module")
def ds(load_or_skip):
    full = load_or_skip("egger_koethenbuerger")
    # Protocol restriction (see module docstring): fresh Dataset on the
    # filtered frame, same spec, so Z/Z_std are recomputed on the subsample.
    df = full.df[full.df["wpop"] < POP_CAP].reset_index(drop=True)
    sub = Dataset(df, full.spec)
    assert sub.n == 42005
    return sub


@pytest.fixture(scope="module")
def found_thresholds(ds):
    """One scan shared by both tests: thresholds with a top-15 cluster rep within 15%."""
    ctf = coarse_to_fine_scan(ds, k=100, n_coarse=3000, top_m=30, rng=np.random.default_rng(0))
    assert ctf.result.model == "normal"  # rcsize in {8,...,24}: integer, not binary
    # 15% multiplicative clustering: |pop_i/pop_j - 1| <~ 0.15 is additive
    # tol = log(1.15) on the log_pop scale (ds.Z is 1-D log_pop), so the
    # tolerance tracks each threshold's own scale — a single raw-pop tol
    # cannot represent "15%" at both 1001 and 10001 simultaneously.
    clusters = cluster_discoveries(ctf.result, ds.Z, tol=np.log(1.0 + REL_TOL), top=50)
    pops = [float(np.exp(c.center_z[0])) for c in clusters[:15]]
    return [t for t in THRESHOLDS if any(abs(p - t) / t < REL_TOL for p in pops)]


def test_multi_threshold_discovery(found_thresholds):
    """Stretch goal (spec section 10): >= 2 statutory thresholds among the
    top-15 clusters. Passes strict (observed: 4 of 5; see module docstring),
    so the plan's conditional xfail decorator is not applied."""
    assert len(found_thresholds) >= 2, f"only found {found_thresholds} of {THRESHOLDS}"


def test_at_least_one_threshold_strict(found_thresholds):
    """Non-stretch floor: the design must find at least the strongest
    statutory jump or the backtest genuinely fails (never xfail)."""
    assert len(found_thresholds) >= 1, f"no statutory threshold found; {THRESHOLDS}"
