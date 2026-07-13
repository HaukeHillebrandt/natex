"""Tests for the panel validation battery (phase-3 task 7).

Covers audit item 1 (+1-rank parametric-bootstrap p-values, per-replica
background refits), item 2 (direct Bernoulli(p_hat) null draws), and item 18
(dependence-preserving ar1_unit nulls; composition/anticipation checks
replacing the information-free McCrary on calendar time), plus power/null
calibration and determinism.
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic_did import make_did_synthetic
from natex.did.background import fit_did_background
from natex.did.panel import CategoricalPanel, build_panel
from natex.did.suddds import DiDDiscovery, SuDDDSResult, suddds_scan
from natex.validate.panel import (
    anticipation_test,
    composition_test,
    panel_randomization_test,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# task-6 synthetic config (tests/test_did_benchmarks_small.py) with restarts
# lowered to 4: replicas rerun the full scan Q times, and restarts=4 recovers
# the planted optimum on every calibration seed while halving the runtime.
CFG = dict(n=1500, d=3, V=4, periods=10, tau=10.0, s_dims=2, s_values=2)
SCAN = dict(windows=(3.0,), restarts=4, bins=4, method="single_delta")


def planted_frame(seed: int, n: int = 500, zeta: float = 8.0, binary: bool = False):
    """Small planted panel (as in test_suddds): step zeta on d0 in {1, 3} at t >= 5."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=(n, 2))
    t = rng.integers(0, 10, size=n).astype(float)
    truth = np.isin(codes[:, 0], [1, 3])
    post = t >= 5.0
    if binary:
        p = 0.2 + 0.6 * (truth & post)
        theta = (rng.random(n) < p).astype(float)
    else:
        theta = rng.normal(0.0, 1.0, size=n) + zeta * (truth & post)
    df = pd.DataFrame(
        {"d0": codes[:, 0], "d1": codes[:, 1], "time": t, "theta": theta}
    )
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["d0", "d1"], time="time"
    )
    return Dataset(df, spec)


def ar1_dataset(seed: int, phi: float = 0.8, n_units: int = 8, T: int = 20, rep: int = 2):
    """AR(1) unit-level panel with NO jump: theta_ut = alpha_u + AR1(phi) + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        alpha = rng.normal(0.0, 1.0)
        e = np.empty(T)
        e[0] = rng.normal(0.0, 1.0)
        for k in range(1, T):
            e[k] = phi * e[k - 1] + rng.normal(0.0, np.sqrt(1.0 - phi**2))
        for t in range(T):
            for _ in range(rep):
                g = int(rng.integers(0, 3))
                rows.append((u, float(t), g, alpha + e[t] + 0.2 * rng.normal()))
    df = pd.DataFrame(rows, columns=["unit", "time", "g", "theta"])
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["g"],
        time="time", unit="unit",
    )
    return Dataset(df, spec)


def trend_panel(
    seed: int,
    jump_at: float | None = None,
    jump: float = 4.0,
    n_units: int = 6,
    T: int = 10,
    rep: int = 3,
    noise: float = 0.5,
):
    """Panel with unit effects + linear trend (+ optional persistent pre-jump)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        a = rng.normal(0.0, 1.0)
        for t in range(T):
            for _ in range(rep):
                g = int(rng.integers(0, 2))
                th = a + 0.3 * t + noise * rng.normal()
                if jump_at is not None and t >= jump_at:
                    th += jump
                rows.append((u, float(t), g, th))
    df = pd.DataFrame(rows, columns=["unit", "time", "g", "theta"])
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["g"],
        time="time", unit="unit",
    )
    panel = build_panel(Dataset(df, spec))
    background = fit_did_background(panel, model="normal")
    return panel, background


def counts_panel(unit: np.ndarray, t: np.ndarray, codes: np.ndarray | None = None):
    """Bare CategoricalPanel for the composition test (theta irrelevant)."""
    n = t.size
    if codes is None:
        codes = np.zeros((n, 0), dtype=np.int64)
        dim_names: list[str] = []
        dim_values: list[np.ndarray] = []
    else:
        dim_names = [f"d{j}" for j in range(codes.shape[1])]
        dim_values = [np.arange(codes[:, j].max() + 1) for j in range(codes.shape[1])]
    return CategoricalPanel(
        codes=codes.astype(np.int64),
        dim_names=dim_names,
        dim_values=dim_values,
        t=t.astype(float),
        theta=np.zeros(n),
        y=None,
        unit=unit.astype(np.int64),
        unit_values=np.unique(unit),
    )


def make_discovery(n: int, t0: float, window: float, method: str = "greedy",
                   model: str = "normal", mask: np.ndarray | None = None) -> DiDDiscovery:
    return DiDDiscovery(
        subset_values={},
        mask=np.ones(n, dtype=bool) if mask is None else mask,
        t0=t0,
        window=window,
        llr=1.0,
        model=model,
        method=method,
    )


# ---------------------------------------------------------------------------
# panel_randomization_test: power / null calibration
# ---------------------------------------------------------------------------


def test_power_planted_zeta8():
    # Calibration (CFG, SCAN, scan rng 1000, test rng 7, Q=19): data seeds
    # 0, 1, 2, 4 give p = 0.05 (= 1/20, the Q=19 minimum; observed LLR
    # 104-125 vs null maxima 12-17). Seed 3's scan converges to a local
    # optimum (obs 10.7, p = 0.50) — the documented Alg 6 multimodality,
    # not a test failure. Seed 0 pinned (obs 125.1, null max 14.7).
    ds, _ = make_did_synthetic(zeta=8.0, rng=np.random.default_rng(0), **CFG)
    res = suddds_scan(ds, rng=np.random.default_rng(1000), **SCAN)
    rep = panel_randomization_test(
        ds, res, Q=19, rng=np.random.default_rng(7), scan_kwargs=SCAN
    )
    assert rep.null_kind == "ar1_unit"
    assert rep.q == 19 and rep.null_max_llrs.shape == (19,)
    assert rep.p_value <= 0.10


def test_null_calibration_zeta0():
    # Same pipeline on zeta=0 data: the observed max LLR is itself a noise
    # max, so p should be well inside (0, 1). Calibration (CFG, SCAN, scan
    # rng 1000, test rng 7, Q=19): data seeds 0-4 give p = 0.45, 0.20, 0.85,
    # 0.75, 0.65 — all >= 0.2. Seed 0 pinned (p = 0.45, margin over 0.2).
    ds, _ = make_did_synthetic(zeta=0.0, rng=np.random.default_rng(0), **CFG)
    res = suddds_scan(ds, rng=np.random.default_rng(1000), **SCAN)
    rep = panel_randomization_test(
        ds, res, Q=19, rng=np.random.default_rng(7), scan_kwargs=SCAN
    )
    assert rep.p_value >= 0.2


def test_ar1_null_vs_iid_dependence_regression():
    # Audit item 18: on an AR(1) phi=0.8 unit-dependent panel with NO jump,
    # iid replicas break the serial dependence, understate the null max-LLR
    # distribution and yield a smaller (anti-conservative) p than ar1_unit.
    # Calibration (Q=19, scan rng 0, test rng 1): data seeds 0-7 give
    # (p_ar1, p_iid) = (.05,.05), (.35,.05), (.60,.15), (.05,.05),
    # (.95,.95), (.65,.15), (.40,.05), (.80,.45); ar1 null maxima 9.8-14.8
    # vs iid 4.4-6.3 on every seed, so p_ar1 >= p_iid always holds (seeds 0
    # and 3 hit the 1/20 floor for both: their observed drift realization is
    # extreme, exactly the anti-conservative case). Seed 1 pinned
    # (p_ar1 = 0.35 vs p_iid = 0.05).
    ds = ar1_dataset(seed=1)
    scan_kw = dict(windows=(5.0,), restarts=3, bins=4, method="single_delta")
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)
    rep_ar1 = panel_randomization_test(
        ds, res, Q=19, rng=np.random.default_rng(1), scan_kwargs=scan_kw, null="ar1_unit"
    )
    rep_iid = panel_randomization_test(
        ds, res, Q=19, rng=np.random.default_rng(1), scan_kwargs=scan_kw, null="iid"
    )
    assert rep_ar1.null_kind == "ar1_unit" and rep_iid.null_kind == "iid"
    assert rep_ar1.p_value >= 0.1
    assert rep_ar1.p_value >= rep_iid.p_value


# ---------------------------------------------------------------------------
# +1-rank rule, determinism, bernoulli dispatch, argument validation
# ---------------------------------------------------------------------------


def test_plus_one_rank_exact_value():
    # Planted zeta=8 dominates every null replica, so with Q=4 the +1-rank
    # p-value is exactly (1 + 0) / (4 + 1) = 1/5 (audit item 1). Calibration
    # (scan rng 0, test rng 2): data seeds 0-4 all give observed 16.9-96.7
    # strictly above null maxima 12.0-19.8, p = 0.2. Seed 0 pinned
    # (obs 88.4 vs null max 19.8).
    ds = planted_frame(seed=0)
    scan_kw = dict(windows=(3.0,), restarts=3, method="single_delta")
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)
    rep = panel_randomization_test(
        ds, res, Q=4, rng=np.random.default_rng(2), scan_kwargs=scan_kw
    )
    assert np.all(rep.null_max_llrs < rep.observed_max_llr)
    assert rep.p_value == 1.0 / 5.0


def test_determinism_same_seed():
    ds = planted_frame(seed=5, n=300)
    scan_kw = dict(windows=(3.0,), restarts=2, method="greedy")
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)

    def run():
        return panel_randomization_test(
            ds, res, Q=3, rng=np.random.default_rng(11), scan_kwargs=scan_kw
        )

    a, b = run(), run()
    np.testing.assert_array_equal(a.null_max_llrs, b.null_max_llrs)
    assert a.p_value == b.p_value


def test_bernoulli_auto_null():
    # audit item 2: binary theta -> auto null is direct Bernoulli(p_hat).
    ds = planted_frame(seed=9, n=400, binary=True)
    scan_kw = dict(windows=(3.0,), restarts=2, method="greedy")
    res = suddds_scan(ds, rng=np.random.default_rng(1), **scan_kw)
    assert res.model == "bernoulli"
    rep = panel_randomization_test(
        ds, res, Q=5, rng=np.random.default_rng(3), scan_kwargs=scan_kw
    )
    assert rep.null_kind == "bernoulli"
    assert 0.0 < rep.p_value <= 1.0
    assert np.all(np.isfinite(rep.null_max_llrs))


def test_issue_14_degenerate_bernoulli_replica_scores_empty_supremum():
    """Issue #14: with a rare binary treatment, a Bernoulli(p_hat) replica draw
    can be all-zero; the per-replica background refit then crashed inside
    sklearn ('needs samples of at least 2 classes'). A one-class draw admits
    no scoreable split, so it scores 0.0 — the supremum over an empty
    candidate set, the same documented convention as a no-discovery replica —
    and the test completes."""
    theta = np.zeros(12)
    theta[5] = 1.0  # a single treated record: p_hat ~ 1/12 per record
    df = pd.DataFrame(
        {
            "g": [0, 1] * 6,
            "time": np.tile(np.arange(6, dtype=float), 2),
            "theta": theta,
        }
    )
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["g"], time="time"
    )
    ds = Dataset(df, spec)
    scan_kw = dict(windows=(3.0,), restarts=2, method="greedy")
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)
    assert res.model == "bernoulli" and len(res.discoveries) == 1
    # Seed 2 draws an all-zero replica (seeds 2-4 crashed before the fix).
    rep = panel_randomization_test(
        ds, res, Q=1, rng=np.random.default_rng(2), scan_kwargs=scan_kw
    )
    assert rep.null_max_llrs.tolist() == [0.0]
    assert rep.p_value == 0.5  # (1 + 0) / (1 + 1): 0.0 < observed llr


def test_issue_13_replicas_search_the_observed_scan_config():
    # Issue #13: with scan_kwargs omitted, the fitted-null background and
    # every replica scan silently used hardcoded bins=4/dims=None/degree=1
    # regardless of what produced the observed max-LLR — replicas searching a
    # smaller space than the observed scan understate the null maximum and
    # give anti-conservative p-values. Defaults must come from the RESOLVED
    # config recorded on the SuDDDSResult; scan_kwargs stays as an override.
    ds = planted_frame(seed=0)
    scan_kw = dict(windows=(3.0,), restarts=2, method="greedy", bins=2, degree=0)
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)
    assert (res.bins, res.degree) == (2, 0)
    assert res.dims is None
    assert (res.min_side, res.n_rho, res.exhaustive_max_values) == (3, 10, 12)
    implicit = panel_randomization_test(ds, res, Q=9, rng=np.random.default_rng(3))
    explicit = panel_randomization_test(
        ds, res, Q=9, rng=np.random.default_rng(3), scan_kwargs=scan_kw
    )
    np.testing.assert_array_equal(implicit.null_max_llrs, explicit.null_max_llrs)
    assert implicit.p_value == explicit.p_value


def test_randomization_invalid_arguments():
    ds = planted_frame(seed=5, n=200)
    scan_kw = dict(windows=(3.0,), restarts=2, method="greedy")
    res = suddds_scan(ds, rng=np.random.default_rng(0), **scan_kw)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        panel_randomization_test(ds, res, Q=19, rng=None, scan_kwargs=scan_kw)
    with pytest.raises(ValueError):
        panel_randomization_test(ds, res, Q=0, rng=rng, scan_kwargs=scan_kw)
    with pytest.raises(ValueError):
        panel_randomization_test(ds, res, Q=19, rng=rng, scan_kwargs=scan_kw, null="bogus")
    with pytest.raises(ValueError):  # bernoulli null on a normal-model result
        panel_randomization_test(ds, res, Q=19, rng=rng, scan_kwargs=scan_kw, null="bernoulli")
    empty = SuDDDSResult(
        discoveries=[], model="normal", method="greedy", windows=(3.0,), restarts=1
    )
    with pytest.raises(ValueError):
        panel_randomization_test(ds, empty, Q=19, rng=rng, scan_kwargs=scan_kw)
    fake_bern = SuDDDSResult(
        discoveries=[make_discovery(ds.n, 5.0, 3.0, model="bernoulli")],
        model="bernoulli", method="greedy", windows=(3.0,), restarts=1,
    )
    with pytest.raises(ValueError):  # dependence-preserving null needs the normal model
        panel_randomization_test(ds, fake_bern, Q=19, rng=rng, null="ar1_unit")


def test_ar1_unit_requires_two_units():
    # spec.unit set but constant -> a single unit; unit-level draws undefined.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g": rng.integers(0, 3, size=80),
            "time": np.tile(np.arange(10.0), 8),
            "theta": rng.normal(size=80),
            "unit": np.zeros(80, dtype=int),
        }
    )
    spec = DatasetSpec(
        treatment="theta", outcome=None, forcing=[], covariates=["g"],
        time="time", unit="unit",
    )
    ds = Dataset(df, spec)
    fake = SuDDDSResult(
        discoveries=[make_discovery(ds.n, 5.0, 3.0)],
        model="normal", method="greedy", windows=(3.0,), restarts=1,
    )
    with pytest.raises(ValueError):
        panel_randomization_test(ds, fake, Q=2, rng=np.random.default_rng(1))


# ---------------------------------------------------------------------------
# composition_test (audit 18 replacement for McCrary-on-time)
# ---------------------------------------------------------------------------


def _balanced(n_units: int = 10, T: int = 10, rep: int = 3):
    unit = np.repeat(np.arange(n_units), T * rep)
    t = np.tile(np.repeat(np.arange(T, dtype=float), rep), n_units)
    return unit, t


def test_composition_balanced_passes():
    unit, t = _balanced()
    panel = counts_panel(unit, t)
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=3.0))
    assert rep.p_value >= 0.99
    assert rep.passed
    assert rep.table.shape == (10, 2)
    assert np.all(rep.table.sum(axis=1) > 0)


def test_composition_attrition_fails():
    # Half the units stop reporting at T0: their post-window counts are 0.
    unit, t = _balanced()
    keep = ~((unit >= 5) & (t >= 5.0))
    panel = counts_panel(unit[keep], t[keep])
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=3.0))
    assert rep.p_value <= 0.01
    assert not rep.passed


def test_composition_by_profile():
    unit, t = _balanced(n_units=4)
    codes = (unit[:, None] % 2).astype(np.int64)  # two profiles
    panel = counts_panel(unit, t, codes=codes)
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=3.0), by="profile")
    assert rep.table.shape == (2, 2)
    assert rep.passed


def test_issue_16_composition_counts_restricted_to_discovery_mask():
    # Issue #16: counts must be taken INSIDE the discovery's subset mask.
    # Masked units 0-1 are perfectly balanced pre/post; out-of-mask units 2-11
    # exist pre-period only. The discovery is internally stable, so the test
    # must pass, with exactly the p-value of the mask-only panel.
    unit_m, t_m = _balanced(n_units=2)
    unit_o, t_o = _balanced(n_units=10)
    keep_o = t_o < 5.0  # out-of-mask units: pre-only (total post attrition)
    unit = np.concatenate([unit_m, unit_o[keep_o] + 2])
    t = np.concatenate([t_m, t_o[keep_o]])
    panel = counts_panel(unit, t)
    mask = unit < 2
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=5.0, mask=mask))
    assert rep.table.shape == (2, 2)
    assert rep.passed
    panel_only = counts_panel(unit_m, t_m)
    rep_only = composition_test(panel_only, make_discovery(panel_only.n, t0=5.0, window=5.0))
    assert rep.p_value == rep_only.p_value


def test_issue_16_composition_masked_subgroup_attrition_detected():
    # Issue #16 (dilution direction): one of two masked units vanishes post.
    # Stable out-of-mask units must not dilute the signal via df inflation.
    unit_m, t_m = _balanced(n_units=2)
    keep_m = ~((unit_m == 1) & (t_m >= 5.0))  # masked unit 1: no post records
    unit_o, t_o = _balanced(n_units=50)
    unit = np.concatenate([unit_m[keep_m], unit_o + 2])
    t = np.concatenate([t_m[keep_m], t_o])
    panel = counts_panel(unit, t)
    mask = unit < 2
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=5.0, mask=mask))
    assert rep.p_value <= 0.01
    assert not rep.passed


def test_issue_16_composition_single_masked_unit_total_attrition_is_degenerate():
    # Issue #16: a single masked unit with total post attrition leaves one
    # usable row (and an empty post column) inside the mask -> NaN, failed —
    # never a silent pass borrowed from balanced out-of-mask units.
    unit_m, t_m = _balanced(n_units=1)
    keep_m = t_m < 5.0
    unit_o, t_o = _balanced(n_units=50)
    unit = np.concatenate([unit_m[keep_m], unit_o + 1])
    t = np.concatenate([t_m[keep_m], t_o])
    panel = counts_panel(unit, t)
    mask = unit < 1
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=5.0, mask=mask))
    assert np.isnan(rep.p_value)
    assert not rep.passed


def test_composition_degenerate_is_nan_never_ok():
    # single usable row -> NaN p, failed (never silently ok)
    unit, t = _balanced(n_units=1)
    panel = counts_panel(unit, t)
    rep = composition_test(panel, make_discovery(panel.n, t0=5.0, window=3.0))
    assert np.isnan(rep.p_value)
    assert not rep.passed
    # empty post column (window beyond the data) -> NaN, failed
    unit, t = _balanced(n_units=4)
    panel = counts_panel(unit, t)
    rep = composition_test(panel, make_discovery(panel.n, t0=12.0, window=3.0))
    assert np.isnan(rep.p_value)
    assert not rep.passed
    with pytest.raises(ValueError):
        composition_test(panel, make_discovery(panel.n, 5.0, 3.0), by="bogus")


# ---------------------------------------------------------------------------
# anticipation_test (audit 18: pre-period placebo jumps, Holm-corrected)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["greedy", "single_delta"])
def test_anticipation_smooth_pre_period_passes(method):
    # Calibration (trend_panel, t0=6, W=2, shifts (1,2,3)): min Holm p over
    # seeds 0-4 is 0.12, 1.00, 0.96, 0.60, 0.98 (worst of the two
    # estimators) — all pass 0.05; the planted-jump variant fails with Holm
    # p < 1e-3 on every seed. Seed 0 pinned (min Holm p 0.12).
    panel, background = trend_panel(seed=0)
    disc = make_discovery(panel.n, t0=6.0, window=2.0, method=method)
    rep = anticipation_test(panel, background, disc)
    assert rep.shifts == (1, 2, 3)
    usable = ~np.isnan(rep.p_holm)
    assert usable.any()
    assert np.all(rep.p_holm[usable] > 0.05)
    assert rep.passed


@pytest.mark.parametrize("method", ["greedy", "single_delta"])
def test_anticipation_pre_jump_fails(method):
    # persistent +4 jump planted at t = 4 = T0 - 2*step, well inside pre-data
    panel, background = trend_panel(seed=0, jump_at=4.0)
    disc = make_discovery(panel.n, t0=6.0, window=2.0, method=method)
    rep = anticipation_test(panel, background, disc)
    usable = ~np.isnan(rep.p_holm)
    assert np.min(rep.p_holm[usable]) <= 0.05
    assert not rep.passed


def test_anticipation_insufficient_support_is_nan():
    # shift=6 puts the placebo cutoff at t=0: the pre side is empty -> NaN p,
    # excluded from Holm; with no usable shift the report fails (never
    # silently ok).
    panel, background = trend_panel(seed=0)
    disc = make_discovery(panel.n, t0=6.0, window=2.0)
    rep = anticipation_test(panel, background, disc, shifts=(6,))
    assert np.isnan(rep.p_values[0]) and np.isnan(rep.p_holm[0])
    assert np.isnan(rep.estimates[0])
    assert not rep.passed


def test_anticipation_invalid_arguments():
    panel, background = trend_panel(seed=0)
    disc = make_discovery(panel.n, t0=6.0, window=2.0)
    with pytest.raises(ValueError):
        anticipation_test(panel, background, disc, shifts=())
    with pytest.raises(ValueError):
        anticipation_test(panel, background, disc, shifts=(0,))
    with pytest.raises(ValueError):
        anticipation_test(panel, background, disc, alpha=0.0)
