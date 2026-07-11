"""Tests for the SuDDDS driver (Algorithms 6-7) — phase 3, task 5.

Covers the audit-11 repair (global incumbent across windows AND restarts),
the audit-12 repair (minimum two-sided support for every cutoff candidate),
y-blindness of the whole scan, planted (t0, subset) recovery through all
three methods, model auto-dispatch (audit 19), and determinism.
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.did.statistics import double_beta_llr_masks, window_stats
from natex.did.suddds import default_windows, optimize_t0, suddds_scan

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def planted_frame(seed: int, n: int = 1500, zeta: float = 8.0, binary: bool = False):
    """m=3 dims x V=4 values, T=10 periods; persistent step zeta on dim0 in {1, 3} at t >= 5.

    The step is persistent (standard DD adoption), so T0 = 5 is the ONLY
    treatment discontinuity — a transient bump would plant a second, equally
    real cutoff at its offset.
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=(n, 3))
    t = rng.integers(0, 10, size=n).astype(float)
    truth = np.isin(codes[:, 0], [1, 3])
    post = t >= 5.0
    if binary:
        p = 0.2 + 0.6 * (truth & post)
        theta = (rng.random(n) < p).astype(float)
    else:
        theta = rng.normal(0.0, 1.0, size=n) + zeta * (truth & post)
    df = pd.DataFrame(
        {
            "d0": codes[:, 0],
            "d1": codes[:, 1],
            "d2": codes[:, 2],
            "time": t,
            "theta": theta,
        }
    )
    return df, truth


def make_dataset(df: pd.DataFrame, outcome: str | None = None) -> Dataset:
    spec = DatasetSpec(
        treatment="theta",
        outcome=outcome,
        forcing=[],
        covariates=["d0", "d1", "d2"],
        time="time",
    )
    return Dataset(df, spec)


def f_score(found: np.ndarray, truth: np.ndarray) -> float:
    tp = float(np.sum(found & truth))
    if tp == 0.0:
        return 0.0
    precision = tp / float(np.sum(found))
    recall = tp / float(np.sum(truth))
    return 2.0 * precision * recall / (precision + recall)


def fingerprint(result):
    """Bitwise-comparable summary of every discovery."""
    return [
        (
            d.llr,
            d.t0,
            d.window,
            d.model,
            d.method,
            d.mask.tobytes(),
            tuple(sorted((k, tuple(v)) for k, v in d.subset_values.items())),
        )
        for d in result.discoveries
    ]


def db_factory(t, r, sigma2, W):
    """make_evaluator closure for optimize_t0: double-beta LLR at each candidate T0."""

    def make(T0):
        ws = window_stats(t, r, sigma2, T0, W)

        def ev(M):
            return double_beta_llr_masks(ws, M)

        return ev, ws

    return make


# ---------------------------------------------------------------------------
# y-blindness: the scan NEVER reads the outcome
# ---------------------------------------------------------------------------


def test_y_blindness_bitwise_identical():
    df, _ = planted_frame(seed=2, n=400)
    df_real = df.assign(y=2.0 * df["theta"].to_numpy() + 0.1)
    df_garbage = df.assign(y=1e12)

    def run(dataset):
        return suddds_scan(
            dataset,
            windows=(3.0,),
            restarts=2,
            rng=np.random.default_rng(11),
        )

    fp_real = fingerprint(run(make_dataset(df_real, outcome="y")))
    fp_garbage = fingerprint(run(make_dataset(df_garbage, outcome="y")))
    fp_none = fingerprint(run(make_dataset(df, outcome=None)))
    assert fp_real == fp_garbage == fp_none
    assert len(fp_real) >= 1  # the scan actually found something to compare


# ---------------------------------------------------------------------------
# planted (t0, subset) recovery for all three methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("greedy", {}),
        ("wcc", {"exhaustive_max_values": 1}),  # force the WCC priority branch
        ("single_delta", {}),
    ],
)
def test_planted_recovery(method, kwargs):
    # Calibration (zeta=8, n=1500, restarts=8, rng seed 0): data seeds
    # 2, 5, 7, 29 recover t0 = 5.0 exactly with F = 1.0 for all three methods;
    # seed 13 + greedy alone converges to a trend-leakage local optimum (the
    # documented Alg 6 multi-modality — restarts=16 or any rng seed 1-4
    # recovers it). Seed 7 pinned; the 0.9 F threshold carries a wide margin.
    df, truth = planted_frame(seed=7)
    result = suddds_scan(
        make_dataset(df),
        windows=(3.0,),
        restarts=8,
        method=method,
        rng=np.random.default_rng(0),
        **kwargs,
    )
    top = result.discoveries[0]
    assert top.t0 == 5.0
    assert f_score(top.mask, truth) >= 0.9
    assert top.model == "normal"
    assert top.method == method
    assert top.llr > 0.0


# ---------------------------------------------------------------------------
# global incumbent across windows/restarts (audit item 11)
# ---------------------------------------------------------------------------


def test_global_incumbent_across_windows():
    # W=3 matches the planted step onset; W=2 (processed LAST) truncates the
    # window and scores strictly lower. An implementation that returned the
    # last window's best instead of the global incumbent would fail.
    # Calibration (restarts=8, rng seed 0): data seeds 2, 5, 7, 13, 29 all
    # give top window 3.0, t0 5.0, and a strictly weaker W=2-only optimum
    # (llr ~262-290 vs ~201-220); seed 7 pinned.
    df, truth = planted_frame(seed=7)
    both = suddds_scan(
        make_dataset(df),
        windows=(3.0, 2.0),
        restarts=8,
        rng=np.random.default_rng(0),
    )
    only_small = suddds_scan(
        make_dataset(df),
        windows=(2.0,),
        restarts=8,
        rng=np.random.default_rng(0),
    )
    top = both.discoveries[0]
    assert top.window == 3.0
    assert top.t0 == 5.0
    # [0] is the incumbent: the max over ALL recorded local optima
    assert top.llr == max(d.llr for d in both.discoveries)
    assert top.llr > only_small.discoveries[0].llr
    # ranked descending, deduped on (mask, t0)
    llrs = [d.llr for d in both.discoveries]
    assert llrs == sorted(llrs, reverse=True)
    keys = [(d.mask.tobytes(), d.t0) for d in both.discoveries]
    assert len(keys) == len(set(keys))
    assert both.top(1) == [top]


# ---------------------------------------------------------------------------
# minimum two-sided support (audit item 12)
# ---------------------------------------------------------------------------


def test_optimize_t0_min_side():
    # candidates {4, 5}: T0=4 has an empty pre side; T0=5 has 2 pre records.
    t = np.array([4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    r = np.array([0.1, -0.2, 3.0, 3.1, 2.9, 3.2, 3.0])
    sigma2 = np.ones_like(t)
    mask = np.ones(t.size, dtype=bool)
    make = db_factory(t, r, sigma2, W=2.0)
    assert optimize_t0(make, t, mask, W=2.0, min_side=3) is None
    got = optimize_t0(make, t, mask, W=2.0, min_side=2)
    assert got is not None
    t0, llr = got
    assert t0 == 5.0
    assert np.isfinite(llr) and llr > 0.0


def test_optimize_t0_candidates_restricted_to_subset():
    t = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    r = np.zeros_like(t)
    mask = np.array([True, True, True, True, False])  # 9.0 is outside the subset
    make = db_factory(t, r, np.ones_like(t), W=10.0)
    got = optimize_t0(make, t, mask, W=10.0, min_side=2)
    assert got is not None
    assert got[0] in {1.0, 2.0, 3.0, 4.0}  # never the out-of-subset time


def test_scan_terminates_when_no_cutoff_qualifies():
    # Every candidate has an empty side within W=1: t in {0, 5} only.
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame(
        {
            "d0": rng.integers(0, 3, size=n),
            "time": np.repeat([0.0, 5.0], n // 2),
            "theta": rng.normal(0.0, 1.0, size=n),
        }
    )
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["d0"], time="time"
    )
    result = suddds_scan(
        Dataset(df, spec), windows=(1.0,), restarts=3, rng=np.random.default_rng(0)
    )
    assert result.discoveries == []  # incumbent stays empty; no exception, no score


# ---------------------------------------------------------------------------
# determinism and rng policy
# ---------------------------------------------------------------------------


def test_determinism_same_seed():
    df, _ = planted_frame(seed=5, n=400)

    def run(seed):
        return suddds_scan(
            make_dataset(df),
            windows=(3.0, 2.0),
            restarts=3,
            rng=np.random.default_rng(seed),
        )

    assert fingerprint(run(4)) == fingerprint(run(4))


def test_rng_required():
    df, _ = planted_frame(seed=5, n=100)
    with pytest.raises(ValueError):
        suddds_scan(make_dataset(df), windows=(3.0,), rng=None)


# ---------------------------------------------------------------------------
# model auto-dispatch (audit item 19)
# ---------------------------------------------------------------------------


def test_auto_model_dispatch_binary():
    df, _ = planted_frame(seed=9, n=300, binary=True)
    result = suddds_scan(
        make_dataset(df),
        windows=(3.0,),
        restarts=2,
        method="greedy",
        rng=np.random.default_rng(1),
    )
    assert result.model == "bernoulli"


def test_auto_model_dispatch_continuous():
    df, _ = planted_frame(seed=9, n=300)
    result = suddds_scan(
        make_dataset(df),
        windows=(3.0,),
        restarts=2,
        method="greedy",
        rng=np.random.default_rng(1),
    )
    assert result.model == "normal"


def test_forced_normal_on_binary_theta_allowed():
    # thesis-parity path: traffic-stop counts were forced through a Normal model
    df, _ = planted_frame(seed=9, n=300, binary=True)
    result = suddds_scan(
        make_dataset(df),
        windows=(3.0,),
        restarts=2,
        model="normal",
        rng=np.random.default_rng(1),
    )
    assert result.model == "normal"


def test_single_delta_requires_normal_model():
    df, _ = planted_frame(seed=9, n=300, binary=True)
    with pytest.raises(ValueError):
        suddds_scan(
            make_dataset(df),
            windows=(3.0,),
            method="single_delta",
            model="bernoulli",
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError):  # auto-resolves to bernoulli -> still raises
        suddds_scan(
            make_dataset(df),
            windows=(3.0,),
            method="single_delta",
            model="auto",
            rng=np.random.default_rng(1),
        )


def test_bernoulli_model_requires_binary_theta():
    df, _ = planted_frame(seed=9, n=300)
    with pytest.raises(ValueError):
        suddds_scan(
            make_dataset(df),
            windows=(3.0,),
            model="bernoulli",
            method="greedy",
            rng=np.random.default_rng(1),
        )


# ---------------------------------------------------------------------------
# default window grid (spec section 10: unreported hyperparameter)
# ---------------------------------------------------------------------------


def test_default_windows_grid():
    t = np.tile(np.arange(10.0), 5)
    # span=9, step=1: (9/8, 9/4, 9/2) -> snapped up, each >= 2*step -> (2, 3, 5)
    assert default_windows(t) == (2.0, 3.0, 5.0)


def test_default_windows_needs_time_variation():
    with pytest.raises(ValueError):
        default_windows(np.zeros(10))


def test_invalid_arguments():
    df, _ = planted_frame(seed=5, n=100)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        suddds_scan(make_dataset(df), windows=(3.0,), method="bogus", rng=rng)
    with pytest.raises(ValueError):
        suddds_scan(make_dataset(df), windows=(3.0,), restarts=0, rng=rng)
    with pytest.raises(ValueError):
        suddds_scan(make_dataset(df), windows=(-1.0,), rng=rng)
    with pytest.raises(ValueError):
        suddds_scan(make_dataset(df), windows=(3.0,), model="bogus", rng=rng)
