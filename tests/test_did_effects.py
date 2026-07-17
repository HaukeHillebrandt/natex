"""Tests for DiD effect estimation (phase 3, task 9).

Covers did_effect (dose normalization, audit item 19), the pluggable
DiDEstimatorBackend protocol, the two-sided studentized tau randomization
test (audit item 5: +1-rank p, matched placebo shapes, negative-effect
regression), and the per-dimension placebo tests with Holm correction.
"""

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.data.synthetic_did import make_did_synthetic
from natex.did import PeriodGaps, period_gaps
from natex.did.controls import dd_control
from natex.did.effects import (
    ESTIMATOR_BACKENDS,
    DiDEffect,
    DiDEstimatorBackend,
    _mean_gap,
    did_effect,
    placebo_dimension_tests,
    tau_randomization_test,
)
from natex.did.panel import CategoricalPanel, build_panel
from natex.did.suddds import DiDDiscovery, suddds_scan

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

# 3 units (a treated, b/c controls) x 4 periods; t0 = 2 (pre: t in {0, 1}).
# Same toy as tests/test_did_controls.py: dd counterfactual post = [12.5, 14.5],
# gess (control b) counterfactual post = [13, 15].
_TOY_Y = {
    0: [10.0, 12.0, 20.0, 22.0],  # a (treated)
    1: [6.0, 8.0, 9.0, 11.0],  # b (parallel to a in pre: gap 4)
    2: [2.0, 2.0, 3.0, 5.0],  # c (not parallel)
}


def toy_panel(drop=(), theta_override=None):
    rows = [(u, tt) for u in range(3) for tt in range(4) if (u, tt) not in drop]
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    if theta_override is None:
        theta = np.array([1.0 if (u == 0 and tt >= 2) else 0.0 for u, tt in rows])
    else:
        theta = np.array([theta_override(u, tt) for u, tt in rows], dtype=float)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array(["a", "b", "c"])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=theta,
        y=np.array([_TOY_Y[u][tt] for u, tt in rows]),
        unit=np.array([u for u, _ in rows], dtype=np.int64),
        unit_values=np.array(["u0", "u1", "u2"]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["a"]},
        mask=codes[:, 0] == 0,
        t0=2.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def profile_panel(
    n_profiles,
    effect,
    seed=0,
    n_t=10,
    t0=5.0,
    n_rep=2,
    noise=0.5,
    post_only=(),
):
    """One categorical dim with ``n_profiles`` values; profile 0 gets ``effect``
    added post-t0. ``post_only`` profiles have no pre-period records (their
    placebo tau is NaN by construction)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=n_profiles)
    rows = [
        (p, tt)
        for p in range(n_profiles)
        for tt in range(n_t)
        if not (p in post_only and tt < t0)
        for _ in range(n_rep)
    ]
    p_arr = np.array([p for p, _ in rows], dtype=np.int64)
    t_arr = np.array([float(tt) for _, tt in rows])
    post = t_arr >= t0
    treated = (p_arr == 0) & post
    y = base[p_arr] + noise * rng.normal(size=len(rows)) + effect * treated
    panel = CategoricalPanel(
        codes=p_arr[:, None].copy(),
        dim_names=["g"],
        dim_values=[np.array([f"g{p}" for p in range(n_profiles)])],
        t=t_arr,
        theta=treated.astype(float),
        y=y,
        unit=p_arr.copy(),
        unit_values=np.array([f"u{p}" for p in range(n_profiles)]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["g0"]},
        mask=p_arr == 0,
        t0=t0,
        window=float(n_t),
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def truth_discovery(ds, truth, bins):
    """Panel + DiDDiscovery from the synthetic DGP's ground truth (true RDiT)."""
    panel = build_panel(ds, bins=bins)
    subset_values = {}
    for j, inc in enumerate(truth.included):
        if not inc.all():
            subset_values[f"x{j}"] = np.arange(1, inc.size + 1)[inc].tolist()
    disc = DiDDiscovery(
        subset_values=subset_values,
        mask=truth.record_mask.copy(),
        t0=truth.t0,
        window=3.0,
        llr=float("nan"),
        model="normal",
        method="single_delta",
    )
    return panel, disc


# ---------------------------------------------------------------------------
# did_effect: hand checks on the task-8 toy panel
# ---------------------------------------------------------------------------


def test_did_effect_toy_dd_hand_check():
    panel, disc = toy_panel()
    eff = did_effect(panel, disc, control="dd")
    assert eff.method == "dd"
    # post gaps: 20 - 12.5 = 7.5 and 22 - 14.5 = 7.5
    np.testing.assert_allclose(eff.tau, 7.5, atol=1e-12)
    np.testing.assert_allclose(eff.se, 0.0, atol=1e-12)  # identical per-period gaps
    assert eff.n_treated_post == 2
    np.testing.assert_allclose(eff.pre_mse, 0.25, atol=1e-12)
    assert eff.dose is None  # binary theta auto-skips normalization (audit 19)
    assert eff.extras["n_skipped_nan"] == 0


def test_did_effect_toy_gess_hand_check():
    panel, disc = toy_panel()
    eff = did_effect(panel, disc, control="gess")
    assert eff.method == "gess"
    # gess selects control b (parallel): post gaps 20 - 13 = 7, 22 - 15 = 7
    np.testing.assert_allclose(eff.tau, 7.0, atol=1e-12)
    np.testing.assert_allclose(eff.pre_mse, 0.0, atol=1e-12)


def test_did_effect_accepts_precomputed_control_result():
    panel, disc = toy_panel()
    ctrl = dd_control(panel, disc)
    eff = did_effect(panel, disc, control=ctrl)
    np.testing.assert_allclose(eff.tau, 7.5, atol=1e-12)
    assert eff.method == "dd"


def test_did_effect_unknown_control_raises():
    panel, disc = toy_panel()
    with pytest.raises(ValueError, match="control"):
        did_effect(panel, disc, control="nope")


def test_did_effect_requires_outcome():
    panel, disc = toy_panel()
    panel.y = None
    with pytest.raises(ValueError, match="outcome"):
        did_effect(panel, disc)


def test_did_effect_nan_counterfactual_cells_skipped_and_counted():
    # Both control records at t=2 removed: dd counterfactual NaN there; the
    # t=3 gap (22 - 14.5 = 7.5) survives. Never NaN -> 0.
    panel, disc = toy_panel(drop={(1, 2), (2, 2)})
    eff = did_effect(panel, disc, control="dd")
    np.testing.assert_allclose(eff.tau, 7.5, atol=1e-12)
    assert eff.n_treated_post == 1
    assert eff.extras["n_skipped_nan"] == 1
    assert np.isnan(eff.se)  # single usable post period: sd undefined, NaN not 0


def test_did_effect_all_nan_counterfactual_is_nan_never_zero():
    panel, _ = toy_panel()
    disc = DiDDiscovery(
        subset_values={},
        mask=np.ones(panel.n, dtype=bool),  # s_tau = D -> empty control
        t0=2.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    eff = did_effect(panel, disc, control="dd")
    assert np.isnan(eff.tau) and np.isnan(eff.se)
    assert eff.n_treated_post == 0


def test_issue_17_mean_gap_se_is_se_of_the_record_weighted_tau():
    # Issue #17: tau is the RECORD-weighted mean over usable post records, so
    # its SE must be the CR1 cluster(=period)-robust SE of that estimator.
    # The old equal-weight std(period_means, ddof=1)/sqrt(h) was invariant to
    # the record weights: two post periods with gaps 0 (n1 records) and 10
    # (1 record) kept se = 5.0 while tau collapsed ~5000x as n1 grew.
    def build(n1):
        t = np.array([0.0, 1.0] + [2.0] * n1 + [3.0])
        n = t.size
        panel = CategoricalPanel(
            codes=np.zeros((n, 1), dtype=np.int64),
            dim_names=["g"],
            dim_values=[np.array(["a"])],
            t=t,
            theta=np.zeros(n),
            y=None,
            unit=np.zeros(n, dtype=np.int64),
            unit_values=np.array(["u0"]),
        )
        disc = DiDDiscovery(
            subset_values={},
            mask=np.ones(n, dtype=bool),
            t0=2.0,
            window=2.0,
            llr=1.0,
            model="normal",
            method="greedy",
        )
        v = np.zeros(n)
        v[-1] = 10.0  # post gaps: n1 zeros at t=2, a single 10 at t=3
        return panel, disc, v, np.zeros(n)

    for n1 in (1, 10, 10000):
        panel, disc, v, v0 = build(n1)
        tau, se, n_used, _n_skipped, h = _mean_gap(panel, disc, v, v0)
        assert (n_used, h) == (n1 + 1, 2)
        N = n1 + 1
        assert tau == pytest.approx(10.0 / N)
        gbar, n_g = np.array([0.0, 10.0]), np.array([float(n1), 1.0])
        expected = float(np.sqrt(2.0 * np.sum(((n_g / N) * (gbar - tau)) ** 2)))
        assert se == pytest.approx(expected)
    # Balanced cells (n1 = 1): the CR1 formula reduces EXACTLY to the old
    # std(period_means, ddof=1)/sqrt(h).
    panel, disc, v, v0 = build(1)
    _tau, se, *_ = _mean_gap(panel, disc, v, v0)
    assert se == pytest.approx(float(np.std([0.0, 10.0], ddof=1) / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# dose normalization (audit item 19)
# ---------------------------------------------------------------------------


def test_dose_normalization_recovers_tau_from_real_theta():
    # Continuous theta: the raw DD contrast estimates zeta*tau = 20, not tau=2.
    # Calibration (seeds 0-7): raw in [18.89, 20.32], dose in [9.70, 10.18],
    # normalized tau in [1.93, 2.05] — every seed clears the bounds below with
    # a wide margin; seed 7 pinned.
    ds, truth = make_did_synthetic(
        n=4000, zeta=10.0, tau=2.0, rng=np.random.default_rng(7)
    )
    panel, disc = truth_discovery(ds, truth, bins=8)
    raw = did_effect(panel, disc, control="dd", dose_normalize=False)
    assert raw.dose is None
    assert abs(raw.tau - 20.0) <= 5.0  # within 25% of zeta*tau
    eff = did_effect(panel, disc, control="dd")  # auto: theta not binary
    assert eff.dose is not None
    assert abs(eff.dose - 10.0) <= 2.5  # theta DD contrast approx zeta
    assert abs(eff.tau - 2.0) <= 0.6


def test_binary_theta_auto_skips_dose_normalization():
    ds, truth = make_did_synthetic(
        n=2000, d=3, V=4, zeta=10.0, tau=10.0, theta_kind="binary",
        rng=np.random.default_rng(0),
    )
    panel, disc = truth_discovery(ds, truth, bins=4)
    eff = did_effect(panel, disc, control="dd")
    assert eff.dose is None
    forced = did_effect(panel, disc, control="dd", dose_normalize=True)
    assert forced.dose is not None


def test_near_zero_dose_gives_nan_never_zero():
    # Constant non-binary theta: dose contrast is exactly 0 -> tau NaN.
    panel, disc = toy_panel(theta_override=lambda u, tt: 0.5)
    eff = did_effect(panel, disc, control="dd")  # auto applies (theta not binary)
    assert eff.dose == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(eff.tau) and np.isnan(eff.se)


def test_dose_normalize_bad_value_raises():
    panel, disc = toy_panel()
    with pytest.raises(ValueError, match="dose_normalize"):
        did_effect(panel, disc, dose_normalize="always")


# ---------------------------------------------------------------------------
# estimator backend protocol (spec non-goal boundary: interface only)
# ---------------------------------------------------------------------------


def test_mean_gap_backend_satisfies_protocol():
    backend = ESTIMATOR_BACKENDS["mean_gap"]
    assert isinstance(backend, DiDEstimatorBackend)
    assert backend.name == "mean_gap"
    panel, disc = toy_panel()
    eff = backend.estimate(panel, disc, dd_control(panel, disc))
    assert isinstance(eff, DiDEffect)
    np.testing.assert_allclose(eff.tau, 7.5, atol=1e-12)


# ---------------------------------------------------------------------------
# tau_randomization_test (audit item 5)
# ---------------------------------------------------------------------------


def test_two_sided_detects_negative_effect():
    # Audit 5 regression: a planted NEGATIVE effect. The two-sided studentized
    # test rejects; the thesis's one-sided upper-tail rank would sit near 1.
    # Calibration (seeds 0-7): observed in [-52.1, -26.1], p = 1/26 = 0.0385
    # and one-sided rank = 1.0 on every seed; seed 1 pinned.
    panel, disc = profile_panel(26, effect=-5.0, seed=1)
    rep = tau_randomization_test(panel, disc, control="dd")
    assert rep.mode == "enumerate"
    assert rep.observed < 0
    assert rep.p_value <= 0.05
    one_sided = (1 + int(np.sum(rep.null_stats >= rep.observed))) / (1 + rep.q)
    assert one_sided > 0.5  # upper-tail rule cannot see this effect


def test_plus_one_rank_rule_exact():
    # 9 usable placebos all strictly below the observed statistic -> p = 1/10.
    panel, disc = profile_panel(10, effect=10.0, seed=2, noise=0.3)
    rep = tau_randomization_test(panel, disc, control="dd")
    assert rep.mode == "enumerate"
    assert rep.q == 9
    assert np.all(np.abs(rep.null_stats) < abs(rep.observed))
    assert rep.p_value == pytest.approx(1.0 / 10.0)


def test_enumerate_mode_is_deterministic_without_rng():
    panel, disc = profile_panel(12, effect=3.0, seed=3)
    rep1 = tau_randomization_test(panel, disc, control="dd")  # rng=None fine
    rep2 = tau_randomization_test(panel, disc, control="dd")
    assert rep1.p_value == rep2.p_value
    np.testing.assert_array_equal(rep1.null_stats, rep2.null_stats)


def test_sample_mode_seeded_determinism_and_rng_required():
    panel, disc = profile_panel(12, effect=3.0, seed=3)
    rep1 = tau_randomization_test(
        panel, disc, control="dd", Q=15, rng=np.random.default_rng(5)
    )
    rep2 = tau_randomization_test(
        panel, disc, control="dd", Q=15, rng=np.random.default_rng(5)
    )
    assert rep1.mode == rep2.mode == "sample"
    assert rep1.p_value == rep2.p_value
    np.testing.assert_array_equal(rep1.null_stats, rep2.null_stats)
    with pytest.raises(ValueError, match="rng"):
        tau_randomization_test(panel, disc, control="dd", Q=15)


def test_auto_switches_to_sampling_when_pool_exceeds_200():
    panel, disc = profile_panel(220, effect=3.0, seed=4, n_rep=1)
    with pytest.raises(ValueError, match="rng"):
        tau_randomization_test(panel, disc, control="dd")
    rep = tau_randomization_test(panel, disc, control="dd", rng=np.random.default_rng(6))
    assert rep.mode == "sample"
    assert rep.q <= 199


def test_nan_placebos_dropped_and_counted():
    # Profiles 6..11 have no pre-period records: their placebo tau is NaN.
    panel, disc = profile_panel(12, effect=6.0, seed=5, post_only=(6, 7, 8, 9, 10, 11))
    rep = tau_randomization_test(panel, disc, control="dd")
    assert rep.extras["n_failed"] == 6
    assert rep.q == 5  # usable placebos
    assert np.isfinite(rep.p_value)


def test_fewer_than_five_usable_placebos_gives_nan_p():
    panel, disc = profile_panel(5, effect=6.0, seed=6)  # pool of 4 placebos
    rep = tau_randomization_test(panel, disc, control="dd")
    assert rep.q == 4
    assert np.isnan(rep.p_value)  # never a fake 1.0


def test_gess_placebos_match_observed_free_dims():
    # Audit 5 matched shapes, gess seeding regression (prop99 backtest): the
    # observed discovery constrains a strict subset of dims; placebo
    # discoveries must constrain the SAME dims (with each placebo profile's
    # own values), not the full profile. Geometry that proves it: two
    # redundant dims d0 = d1 = unit id, so every full profile is >= 2 value
    # changes away from every other unit and a full-profile-seeded gess can
    # never expand to a nonempty control (38/38 placebos failed on prop99);
    # with d1 left free (as in the observed discovery) one d0 expansion
    # reaches another unit and every placebo is usable.
    rng = np.random.default_rng(7)
    n_u, n_t = 6, 10
    rows = [(u, tt) for u in range(n_u) for tt in range(n_t)]
    u_arr = np.array([u for u, _ in rows], dtype=np.int64)
    t_arr = np.array([float(tt) for _, tt in rows])
    treated = (u_arr == 0) & (t_arr >= 5.0)
    base = rng.normal(0.0, 1.0, size=n_u)
    y = base[u_arr] + 0.3 * rng.normal(size=len(rows)) + 6.0 * treated
    panel = CategoricalPanel(
        codes=np.column_stack([u_arr, u_arr]).astype(np.int64),
        dim_names=["d0", "d1"],
        dim_values=[
            np.array([f"a{u}" for u in range(n_u)]),
            np.array([f"b{u}" for u in range(n_u)]),
        ],
        t=t_arr,
        theta=treated.astype(float),
        y=y,
        unit=u_arr.copy(),
        unit_values=np.array([f"u{u}" for u in range(n_u)]),
    )
    disc = DiDDiscovery(
        subset_values={"d0": ["a0"]},  # d1 free, exactly as a scan can leave it
        mask=u_arr == 0,
        t0=5.0,
        window=5.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    rep = tau_randomization_test(panel, disc, control="gess")
    assert rep.mode == "enumerate"
    assert rep.extras["n_failed"] == 0
    assert rep.q == 5  # every placebo produced a usable gess control
    assert np.isfinite(rep.p_value)


# ---------------------------------------------------------------------------
# placebo_dimension_tests
# ---------------------------------------------------------------------------


def two_dim_panel(jump_h=False, n_g=10, n_t=10, t0=5.0, n_rep=2, seed=8):
    """Dims g (defines s_tau = {g0}) and h (free). ``jump_h=True`` plants a
    composition jump: every (g0, post) record has h = 1."""
    rng = np.random.default_rng(seed)
    rows = [(g, tt) for g in range(n_g) for tt in range(n_t) for _ in range(n_rep)]
    g_arr = np.array([g for g, _ in rows], dtype=np.int64)
    t_arr = np.array([float(tt) for _, tt in rows])
    h_arr = rng.integers(0, 2, size=len(rows))
    post = t_arr >= t0
    treated = (g_arr == 0) & post
    if jump_h:
        h_arr = np.where(treated, 1, h_arr)
    base = rng.normal(0.0, 1.0, size=n_g)
    y = base[g_arr] + 0.5 * rng.normal(size=len(rows)) - 4.0 * treated
    panel = CategoricalPanel(
        codes=np.column_stack([g_arr, h_arr]).astype(np.int64),
        dim_names=["g", "h"],
        dim_values=[np.array([f"g{g}" for g in range(n_g)]), np.array([0, 1])],
        t=t_arr,
        theta=treated.astype(float),
        y=y,
        unit=(2 * g_arr + h_arr).astype(np.int64),
        unit_values=np.array([f"u{k}" for k in range(2 * n_g)]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["g0"]},
        mask=g_arr == 0,
        t0=t0,
        window=float(n_t),
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def test_placebo_dimensions_vacuous_when_all_dims_define_s_tau():
    panel, disc = profile_panel(12, effect=3.0, seed=3)
    rep = placebo_dimension_tests(panel, disc, control="dd")
    assert rep.p_values == {} and rep.p_holm == {}
    assert rep.passed is True
    assert rep.note is not None


def test_placebo_dimensions_clean_composition_passes():
    # Null calibration (seeds 0-15): p_holm in [0.1, 1.0] (0.1 is the +1-rank
    # floor with the 9-placebo pool), none anti-conservative; seed 8 pinned.
    # With FULL-profile placebos (pre-repair) the null was anti-conservative:
    # 5/16 seeds below 0.06 — the reduced-profile matching fixed it.
    panel, disc = two_dim_panel(jump_h=False)
    rep = placebo_dimension_tests(panel, disc, control="dd")
    assert set(rep.p_values) == {"h"}
    assert rep.p_holm["h"] >= rep.p_values["h"]
    assert rep.p_holm["h"] > 0.05
    assert rep.passed is True


def test_placebo_dimensions_time_invariant_composition_passes():
    # Zero-movement regression (prop99 backtest): when a free dimension is
    # FIXED per unit (h = g % 2 here; prop99's state-level covariates are all
    # time-invariant), its composition share is constant over time, so every
    # gap is exactly 0 with se exactly 0 — proof of NO composition movement,
    # not an estimation failure. The studentized statistic must be 0 (least
    # extreme), giving p = 1 and a pass; mapping 0/0 to NaN failed the whole
    # battery on a panel where a composition jump is impossible.
    rng = np.random.default_rng(11)
    n_g, n_t, n_rep = 10, 10, 2
    rows = [(g, tt) for g in range(n_g) for tt in range(n_t) for _ in range(n_rep)]
    g_arr = np.array([g for g, _ in rows], dtype=np.int64)
    t_arr = np.array([float(tt) for _, tt in rows])
    h_arr = g_arr % 2  # deterministic per g: time-invariant composition
    treated = (g_arr == 0) & (t_arr >= 5.0)
    base = rng.normal(0.0, 1.0, size=n_g)
    y = base[g_arr] + 0.5 * rng.normal(size=len(rows)) - 4.0 * treated
    panel = CategoricalPanel(
        codes=np.column_stack([g_arr, h_arr]).astype(np.int64),
        dim_names=["g", "h"],
        dim_values=[np.array([f"g{g}" for g in range(n_g)]), np.array([0, 1])],
        t=t_arr,
        theta=treated.astype(float),
        y=y,
        unit=g_arr.copy(),
        unit_values=np.array([f"u{g}" for g in range(n_g)]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["g0"]},
        mask=g_arr == 0,
        t0=5.0,
        window=float(n_t),
        llr=1.0,
        model="normal",
        method="greedy",
    )
    rep = placebo_dimension_tests(panel, disc, control="dd")
    assert rep.p_values["h"] == pytest.approx(1.0)
    assert rep.p_holm["h"] == pytest.approx(1.0)
    assert rep.passed is True


def test_placebo_dimensions_object_dtype_dim_values():
    # build_panel stores non-numeric dims as OBJECT-dtype arrays (pd.factorize
    # uniques) whose elements are plain Python scalars without `.item()` — the
    # Epoch dogfood crash (natex-runs/REPORT.md section 3, finding 1), second
    # site: the modal-value decode of each free dimension.
    panel, disc = two_dim_panel(jump_h=False)
    panel.dim_values = [np.asarray(v, dtype=object) for v in panel.dim_values]
    rep = placebo_dimension_tests(panel, disc, control="dd")
    assert set(rep.p_values) == {"h"}
    modal = rep.extras["modal_values"]["h"]
    assert modal in (0, 1) and type(modal) is int


def test_placebo_dimensions_detect_composition_jump():
    # Pool = n_g - 1 = 24 whole-g placebo cells (the tested dim is removed
    # from the profile definition), so the minimum p is 1/25 = 0.04.
    panel, disc = two_dim_panel(jump_h=True, n_g=25)
    rep = placebo_dimension_tests(panel, disc, control="dd")
    assert rep.p_holm["h"] <= 0.05
    assert rep.passed is False


# ---------------------------------------------------------------------------
# period_gaps (phase report-paper, task 2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scanned_synthetic():
    """Seeded end-to-end scan: synthetic DGP -> panel -> top discovery."""
    ds, _truth = make_did_synthetic(
        n=1500, d=2, V=3, zeta=10.0, tau=10.0, rng=np.random.default_rng(0)
    )
    panel = build_panel(ds, bins=3)
    result = suddds_scan(ds, bins=3, panel=panel, rng=np.random.default_rng(0))
    return ds, panel, result.discoveries[0]


class TestPeriodGaps:
    def test_sorted_times_matched_shapes_and_metadata(self, scanned_synthetic):
        _ds, panel, top = scanned_synthetic
        g = period_gaps(panel, top, "dd")
        assert isinstance(g, PeriodGaps)
        assert np.all(np.diff(g.times) > 0)  # sorted, unique
        assert g.times.shape == g.gap.shape == g.n.shape
        assert np.issubdtype(g.n.dtype, np.integer)
        assert np.all(g.n >= 1)  # zero-usable periods are OMITTED, never zero-filled
        assert np.all(np.isfinite(g.gap))
        assert g.t0 == top.t0
        assert g.control == "dd"

    def test_pre_gaps_near_zero_post_gaps_large(self, scanned_synthetic):
        # Raw (un-normalized) gaps: the continuous-theta DGP plants a y jump of
        # zeta * tau = 100 on s_tau post-T0, so post >> pre by construction.
        # Calibration (seeds 0-3): pre mean in [-0.17, 0.09], post mean in
        # [99.5, 103.1] — thresholds stay loose against seed drift; seed 0
        # pinned in the fixture.
        _ds, panel, top = scanned_synthetic
        g = period_gaps(panel, top, "dd")
        pre = g.times < g.t0
        post = ~pre
        assert pre.any() and post.any()
        assert abs(float(np.mean(g.gap[pre]))) < 2.0
        assert float(np.mean(g.gap[post])) > 5.0
        assert float(np.mean(g.gap[post])) > float(np.mean(g.gap[pre])) + 4.0

    def test_post_gap_average_matches_did_effect_tau(self, scanned_synthetic):
        # Same records, same fitted contrast (audit 19): the n-weighted post
        # average equals did_effect's tau (reduced-form tau_rf when the effect
        # is dose-normalized — period_gaps reports raw y gaps).
        _ds, panel, top = scanned_synthetic
        g = period_gaps(panel, top, "dd")
        eff = did_effect(panel, top, "dd")
        post = g.times >= g.t0
        pooled = float(np.average(g.gap[post], weights=g.n[post]))
        target = eff.tau if eff.dose is None else eff.extras["tau_rf"]
        assert pooled == pytest.approx(target, rel=1e-6)
        assert int(g.n[post].sum()) == eff.n_treated_post

    def test_requires_outcome(self, scanned_synthetic):
        # Rebuilt from a no-outcome Dataset: reporting never fabricates y.
        ds, _panel, top = scanned_synthetic
        spec_no_y = ds.spec.model_copy(update={"outcome": None})
        ds_no_y = Dataset(ds.df.drop(columns=["y"]), spec_no_y)
        panel_no_y = build_panel(ds_no_y, bins=3)
        assert panel_no_y.y is None
        with pytest.raises(ValueError, match="outcome"):
            period_gaps(panel_no_y, top, "dd")
