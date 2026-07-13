"""Tests for DiD control identification (phase 3, task 8).

Covers the three control constructions of thesis section 6.3.2 with the audit
repairs: dd_control (Eq 6.18, count-corrected denominators), synthetic_control
(unit-level simplex weights), gess_control (Algorithm 9 with the argmin fix
and MSE = +inf initialization).
"""

import numpy as np
import pytest

from natex.did.controls import dd_control, gess_control, synthetic_control
from natex.did.effects import did_effect
from natex.did.panel import CategoricalPanel
from natex.did.suddds import DiDDiscovery

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

# 3 units (a treated, b/c controls) x 4 periods; t0 = 2 (pre: t in {0, 1}).
_TOY_Y = {
    0: [10.0, 12.0, 20.0, 22.0],  # a (treated)
    1: [6.0, 8.0, 9.0, 11.0],  # b (parallel to a in pre: gap 4)
    2: [2.0, 2.0, 3.0, 5.0],  # c (not parallel)
}


def toy_panel(drop=(), y_override=None):
    """Toy 1-dim panel; ``drop`` removes (unit, time) records entirely,
    ``y_override`` maps (unit, time) -> y value (e.g. NaN)."""
    rows = [(u, tt) for u in range(3) for tt in range(4) if (u, tt) not in drop]
    y_map = dict(y_override or {})
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array(["a", "b", "c"])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.array([1.0 if (u == 0 and tt >= 2) else 0.0 for u, tt in rows]),
        y=np.array([y_map.get((u, tt), _TOY_Y[u][tt]) for u, tt in rows]),
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


def all_treated_discovery(panel):
    """s_tau = D: every dimension unconstrained, all records treated."""
    return DiDDiscovery(
        subset_values={},
        mask=np.ones(panel.n, dtype=bool),
        t0=2.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )


# ---------------------------------------------------------------------------
# dd_control
# ---------------------------------------------------------------------------


def test_dd_control_hand_check():
    panel, disc = toy_panel()
    res = dd_control(panel, disc)
    assert res.method == "dd"
    # alpha = mean(treated pre) - mean(control pre) = 11 - 4.5
    np.testing.assert_allclose(res.alpha, 6.5, atol=1e-12)
    # per-time control means: 4, 5, 6, 8 -> y0 = alpha + mean
    np.testing.assert_allclose(res.y0_hat, [10.5, 11.5, 12.5, 14.5], atol=1e-12)
    # pre residuals -0.5, +0.5 -> mean square 0.25
    np.testing.assert_allclose(res.pre_mse, 0.25, atol=1e-12)
    np.testing.assert_array_equal(res.control_mask, panel.codes[:, 0] != 0)
    assert res.weights is None
    assert res.extras["n_undefined_times"] == 0


def test_dd_control_unbalanced_counts():
    # Drop control record (unit c, t=1): denominators must follow the records
    # actually present (audit typo repair), not |D \ s_tau| or T0-scaled counts.
    panel, disc = toy_panel(drop={(2, 1)})
    res = dd_control(panel, disc)
    alpha = 11.0 - (6.0 + 2.0 + 8.0) / 3.0  # control pre now has 3 records
    np.testing.assert_allclose(res.alpha, alpha, rtol=1e-12)
    expected_y0 = np.array([alpha + 4.0, alpha + 8.0, alpha + 6.0, alpha + 8.0])
    np.testing.assert_allclose(res.y0_hat, expected_y0, rtol=1e-12)
    expected_mse = ((10.0 - expected_y0[0]) ** 2 + (12.0 - expected_y0[1]) ** 2) / 2.0
    np.testing.assert_allclose(res.pre_mse, expected_mse, rtol=1e-12)
    assert res.extras["n_undefined_times"] == 0


def test_dd_control_nan_y_excluded_like_missing_record():
    # A control record with NaN y behaves as if absent (count-corrected).
    panel, disc = toy_panel(y_override={(2, 1): np.nan})
    res = dd_control(panel, disc)
    alpha = 11.0 - (6.0 + 2.0 + 8.0) / 3.0
    np.testing.assert_allclose(res.alpha, alpha, rtol=1e-12)


def test_dd_control_time_without_controls_is_nan():
    # Remove BOTH control records at t=1: counterfactual undefined there (NaN,
    # never 0); pre_mse finite over the remaining defined pre record.
    panel, disc = toy_panel(drop={(1, 1), (2, 1)})
    res = dd_control(panel, disc)
    np.testing.assert_allclose(res.alpha, 7.0, atol=1e-12)  # 11 - (6+2)/2
    np.testing.assert_allclose(res.y0_hat[[0, 2, 3]], [11.0, 13.0, 15.0], atol=1e-12)
    assert np.isnan(res.y0_hat[1])
    np.testing.assert_allclose(res.pre_mse, 1.0, atol=1e-12)  # (10 - 11)^2 only
    assert res.extras["n_undefined_times"] == 1


def test_dd_control_empty_control_edge():
    # s_tau = D -> empty control: pre_mse = +inf, y0_hat all-NaN, no exception.
    panel, _ = toy_panel()
    disc = all_treated_discovery(panel)
    res = dd_control(panel, disc)
    assert res.pre_mse == np.inf
    assert np.all(np.isnan(res.y0_hat))
    assert np.isnan(res.alpha)


def test_dd_control_requires_outcome():
    panel, disc = toy_panel()
    panel.y = None
    with pytest.raises(ValueError, match="outcome"):
        dd_control(panel, disc)


# ---------------------------------------------------------------------------
# synthetic_control
# ---------------------------------------------------------------------------


def synth_panel():
    """4 units x 6 periods; treated unit 0 is exactly 0.3*u1 + 0.7*u2 pre-t0=4."""
    y_by_unit = {
        0: [8.0, 7.6, 7.2, 6.8, 9.4, 9.0],  # 0.3*u1 + 0.7*u2 (+3 effect post)
        1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2: [11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        3: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    }
    rows = [(u, tt) for u in range(4) for tt in range(6)]
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array(["tr", "c1", "c2", "c3"])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.array([1.0 if (u == 0 and tt >= 4) else 0.0 for u, tt in rows]),
        y=np.array([y_by_unit[u][tt] for u, tt in rows]),
        unit=np.array([u for u, _ in rows], dtype=np.int64),
        unit_values=np.array(["u0", "u1", "u2", "u3"]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["tr"]},
        mask=codes[:, 0] == 0,
        t0=4.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def test_synthetic_control_recovers_convex_weights():
    panel, disc = synth_panel()
    res = synthetic_control(panel, disc)
    assert res.method == "synthetic"
    assert res.alpha is None  # weights absorb levels, no offset
    assert list(res.extras["control_units"]) == [1, 2, 3]
    np.testing.assert_allclose(res.weights, [0.3, 0.7, 0.0], atol=0.02)
    assert abs(res.weights.sum() - 1.0) <= 1e-6
    assert res.weights.min() >= -1e-9
    assert res.pre_mse < 1e-6
    # post-period counterfactual: 0.3*[5,6] + 0.7*[7,6] = [6.4, 6.0]
    np.testing.assert_allclose(res.y0_hat[4:], [6.4, 6.0], atol=0.05)


def test_synthetic_control_scale_invariant_optimization():
    # Regression (prop99 backtest, task 12): with ~38 donors and outcomes at
    # the raw-data scale (~100-300), SLSQP on the UNSCALED SSE objective
    # stalls at the uniform start ("success" after ~5 iterations, weights all
    # 1/n_c, SSE ~3000) because its internal accuracy threshold is absolute.
    # The objective must be scale-normalized so the fit is invariant to the
    # outcome's units. Deterministic: seeded donor trajectories, treated unit
    # is exactly 0.6*donor0 + 0.4*donor1, so the optimum has SSE ~ 0.
    rng = np.random.default_rng(0)
    n_pre, n_post, n_c = 19, 6, 38
    n_t = n_pre + n_post
    base = rng.uniform(80.0, 280.0, size=n_c)
    donors = base[None, :] + np.cumsum(rng.normal(0.0, 3.0, size=(n_t, n_c)), axis=0)
    w_true = np.zeros(n_c)
    w_true[0], w_true[1] = 0.6, 0.4
    treated = donors @ w_true  # exact convex combination, pre AND post
    y_by_time = np.column_stack([treated, donors])  # unit 0 treated
    rows = [(u, tt) for u in range(n_c + 1) for tt in range(n_t)]
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array([f"s{u}" for u in range(n_c + 1)])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.array([1.0 if (u == 0 and tt >= n_pre) else 0.0 for u, tt in rows]),
        y=np.array([y_by_time[tt, u] for u, tt in rows]),
        unit=np.array([u for u, _ in rows], dtype=np.int64),
        unit_values=np.array([f"u{u}" for u in range(n_c + 1)]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["s0"]},
        mask=codes[:, 0] == 0,
        t0=float(n_pre),
        window=5.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    res = synthetic_control(panel, disc)
    assert res.pre_mse < 1e-6  # stalled-at-uniform fit has pre_mse ~ 150
    np.testing.assert_allclose(res.weights[:2], [0.6, 0.4], atol=0.01)
    assert float(res.weights[2:].max()) <= 0.01


def test_synthetic_control_drops_incomplete_donors_instead_of_failing():
    # A donor unit missing one pre-period record is excluded from the pool
    # (Abadie's balanced-donor requirement); the fit proceeds on the rest.
    # Before the repair one sparse unit voided every common pre time -> NaN.
    panel, disc = synth_panel()
    keep = ~((panel.unit == 3) & (panel.t == 1.0))
    sparse = CategoricalPanel(
        codes=panel.codes[keep],
        dim_names=panel.dim_names,
        dim_values=panel.dim_values,
        t=panel.t[keep],
        theta=panel.theta[keep],
        y=panel.y[keep],
        unit=panel.unit[keep],
        unit_values=panel.unit_values,
    )
    disc = DiDDiscovery(
        subset_values=disc.subset_values,
        mask=sparse.codes[:, 0] == 0,
        t0=4.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    res = synthetic_control(sparse, disc)
    assert list(res.extras["control_units"]) == [1, 2]
    assert res.extras["n_donors_dropped"] == 1
    np.testing.assert_allclose(res.weights, [0.3, 0.7], atol=0.02)
    assert res.pre_mse < 1e-6


def test_synthetic_control_renormalizes_small_missing_weight():
    # Treated = 0.05*u1 + 0.95*u2 in the pre period. Dropping u1's record at a
    # post time leaves missing weight ~0.05 <= 0.1: the time stays defined via
    # renormalization over present donors (y0 ~= u2's value), never NaN-voided.
    y_by_unit = {
        0: [1.5, 2.4, 3.3, 4.2, 8.0, 8.0],  # 0.05*u1 + 0.95*u2 pre-t0=4
        1: [11.0, 10.0, 9.0, 8.0, 7.0, 6.0],
        2: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3: [30.0, 20.0, 35.0, 25.0, 30.0, 20.0],  # poor fit: weight ~0
    }
    rows = [(u, tt) for u in range(4) for tt in range(6) if (u, tt) != (1, 5)]
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array(["tr", "c1", "c2", "c3"])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.array([1.0 if (u == 0 and tt >= 4) else 0.0 for u, tt in rows]),
        y=np.array([y_by_unit[u][tt] for u, tt in rows]),
        unit=np.array([u for u, _ in rows], dtype=np.int64),
        unit_values=np.array(["u0", "u1", "u2", "u3"]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["tr"]},
        mask=codes[:, 0] == 0,
        t0=4.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    res = synthetic_control(panel, disc)
    np.testing.assert_allclose(res.weights, [0.05, 0.95, 0.0], atol=0.02)
    # t=5: u1 missing, missing weight ~0.05 -> renormalized to ~u2's value 6.
    assert np.isfinite(res.y0_hat[5])
    np.testing.assert_allclose(res.y0_hat[5], 6.0, atol=0.3)


def test_synthetic_control_no_control_units():
    panel, disc = synth_panel()
    disc = all_treated_discovery(panel)
    res = synthetic_control(panel, disc)
    assert res.pre_mse == np.inf
    assert np.all(np.isnan(res.y0_hat))
    assert res.weights is None


def test_synthetic_control_requires_outcome():
    panel, disc = synth_panel()
    panel.y = None
    with pytest.raises(ValueError, match="outcome"):
        synthetic_control(panel, disc)


# ---------------------------------------------------------------------------
# gess_control
# ---------------------------------------------------------------------------


def test_gess_argmin_regression():
    # Candidate 'b' gives control MSE 0 (parallel), candidate 'c' gives MSE 1.
    # Algorithm 9 as PRINTED (argmax, audit item 14) would select 'c'; the
    # repaired argmin must select 'b' and then stop (adding 'c' raises MSE).
    panel, disc = toy_panel()
    res = gess_control(panel, disc)
    assert res.method == "gess"
    assert res.extras["expansions"] == [{"dim": "g", "value": "b"}]
    np.testing.assert_array_equal(res.control_mask, panel.codes[:, 0] == 1)
    np.testing.assert_allclose(res.alpha, 4.0, atol=1e-12)
    np.testing.assert_allclose(res.y0_hat, [10.0, 12.0, 13.0, 15.0], atol=1e-12)
    np.testing.assert_allclose(res.pre_mse, 0.0, atol=1e-12)
    trace = res.extras["mse_trace"]
    assert np.isinf(trace[0])  # incumbent initialized at +inf (empty control)
    np.testing.assert_allclose(trace[1], 0.0, atol=1e-12)
    assert np.all(np.diff(trace) <= 0)  # monotone nonincreasing accepted steps


def test_gess_monotone_trace_and_termination():
    rng = np.random.default_rng(42)
    n_units, n_t = 6, 6
    rows = [(u, tt) for u in range(n_units) for tt in range(n_t)]
    codes = np.array([[u] for u, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array([f"v{u}" for u in range(n_units)])],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.zeros(len(rows)),
        y=rng.normal(size=len(rows)),
        unit=np.array([u for u, _ in rows], dtype=np.int64),
        unit_values=np.array([f"u{u}" for u in range(n_units)]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["v0"]},
        mask=codes[:, 0] == 0,
        t0=3.0,
        window=3.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    res = gess_control(panel, disc)
    trace = np.asarray(res.extras["mse_trace"])
    assert np.all(np.diff(trace) <= 0)
    assert len(res.extras["expansions"]) <= n_units - 1  # bounded by value count
    assert res.y0_hat.shape == (n_t,)
    assert not np.any(res.control_mask & disc.mask)  # s_c disjoint from s_tau


def two_dim_panel():
    """Dims g in {a,b,c}, h in {x,y}; s_tau = (g=a, h=x); (a,y) is parallel."""
    y_by_profile = {
        ("a", "x"): [10.0, 12.0, 20.0, 22.0],  # treated
        ("a", "y"): [7.0, 9.0, 10.0, 12.0],  # parallel pre (gap 3) -> MSE 0
        ("b", "x"): [0.0, 5.0, 1.0, 2.0],
        ("c", "x"): [9.0, 0.0, 2.0, 3.0],
        ("b", "y"): [1.0, 1.0, 1.0, 1.0],
        ("c", "y"): [2.0, 2.0, 2.0, 2.0],
    }
    g_vals, h_vals = ["a", "b", "c"], ["x", "y"]
    rows = [(g, h, tt) for g in range(3) for h in range(2) for tt in range(4)]
    codes = np.array([[g, h] for g, h, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g", "h"],
        dim_values=[np.array(g_vals), np.array(h_vals)],
        t=np.array([float(tt) for _, _, tt in rows]),
        theta=np.zeros(len(rows)),
        y=np.array([y_by_profile[(g_vals[g], h_vals[h])][tt] for g, h, tt in rows]),
        unit=np.array([2 * g + h for g, h, _ in rows], dtype=np.int64),
        unit_values=np.array([f"{g}{h}" for g in g_vals for h in h_vals]),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["a"], "h": ["x"]},
        mask=(codes[:, 0] == 0) & (codes[:, 1] == 0),
        t0=2.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def test_gess_full_dimension_expands_whole_dims():
    panel, disc = two_dim_panel()
    res = gess_control(panel, disc, full_dimension=True)
    # Relaxing h entirely (control = (a,y), MSE 0) beats relaxing g (MSE > 0).
    assert res.extras["expansions"] == [{"dim": "h", "value": None}]
    # The expanded dim is all-True -> omitted from the final profile.
    assert res.extras["profile"] == {"g": ["a"]}
    expected_ctrl = (panel.codes[:, 0] == 0) & (panel.codes[:, 1] == 1)
    np.testing.assert_array_equal(res.control_mask, expected_ctrl)
    np.testing.assert_allclose(res.pre_mse, 0.0, atol=1e-12)


def test_gess_value_mode_expands_single_values_of_constrained_dims():
    panel, disc = two_dim_panel()
    res = gess_control(panel, disc)
    # Value mode: first expansion adds the single value h='y' (control (a,y)).
    assert res.extras["expansions"][0] == {"dim": "h", "value": "y"}
    assert all(e["value"] is not None for e in res.extras["expansions"])


def test_gess_empty_control_edge():
    # s_tau = D: no constrained dims -> no candidates -> graceful +inf result.
    panel, _ = toy_panel()
    disc = all_treated_discovery(panel)
    res = gess_control(panel, disc)
    assert res.pre_mse == np.inf
    assert np.all(np.isnan(res.y0_hat))
    assert not res.control_mask.any()
    assert res.extras["expansions"] == []
    assert res.extras["mse_trace"] == [np.inf]


def test_gess_requires_outcome():
    panel, disc = toy_panel()
    panel.y = None
    with pytest.raises(ValueError, match="outcome"):
        gess_control(panel, disc)


def _sparse_control_panel(with_complete_control=True):
    """Treated 'tau' at t=0..6 (t0=5); complete control 'A' (small wiggle, so
    pre-MSE 0.0864 > 0); sparse control 'B' with records at t=0,1 ONLY —
    a perfect pre fit after alpha (pre-MSE 0) but ZERO post records."""
    rows = [("tau", tt) for tt in range(7)]
    if with_complete_control:
        rows += [("A", tt) for tt in range(7)]
    rows += [("B", 0), ("B", 1)]
    values = sorted({g for g, _ in rows})  # ["A", "B", "tau"] / ["B", "tau"]
    code_of = {g: j for j, g in enumerate(values)}

    def y_of(g, tt):
        if g == "tau":
            return 10.0 + 5.0 * (tt >= 5)
        if g == "A":
            return 8.0 + (0.3 if tt % 2 == 0 else -0.3)
        return 3.0  # B: constant, perfectly parallel to tau's flat pre period

    codes = np.array([[code_of[g]] for g, _ in rows], dtype=np.int64)
    panel = CategoricalPanel(
        codes=codes,
        dim_names=["g"],
        dim_values=[np.array(values)],
        t=np.array([float(tt) for _, tt in rows]),
        theta=np.zeros(len(rows)),
        y=np.array([y_of(g, tt) for g, tt in rows]),
        unit=codes[:, 0].copy(),
        unit_values=np.array(values),
    )
    disc = DiDDiscovery(
        subset_values={"g": ["tau"]},
        mask=codes[:, 0] == code_of["tau"],
        t0=5.0,
        window=2.0,
        llr=1.0,
        model="normal",
        method="greedy",
    )
    return panel, disc


def test_issue_22_gess_rejects_candidates_without_post_support():
    # Issue #22: the GESS argmin ranked candidates by pre-MSE alone, so the
    # sparse control B (pre-MSE 0, zero post records) beat the complete
    # control A (pre-MSE 0.0864) and the effect degenerated to tau = NaN with
    # a usable control available. A candidate that cannot produce ANY usable
    # post counterfactual is a failure = +inf by the existing MSE policy.
    panel, disc = _sparse_control_panel()
    res = gess_control(panel, disc)
    assert res.extras["expansions"] == [{"dim": "g", "value": "A"}]
    assert np.isfinite(res.pre_mse)
    np.testing.assert_allclose(res.pre_mse, 0.0864, atol=1e-12)
    post = panel.t[np.flatnonzero(disc.mask)] >= disc.t0
    assert np.all(np.isfinite(res.y0_hat[post]))
    eff = did_effect(panel, disc, control=res)
    np.testing.assert_allclose(eff.tau, 5.06, atol=1e-12)
    assert eff.n_treated_post == 2


def test_issue_22_gess_no_post_support_anywhere_returns_empty_control():
    # With ONLY the sparse candidate available no expansion is estimable:
    # +inf never strictly beats the +inf incumbent, so GESS honestly returns
    # the empty control (tau = NaN) instead of a control that cannot produce
    # any post counterfactual.
    panel, disc = _sparse_control_panel(with_complete_control=False)
    res = gess_control(panel, disc)
    assert res.pre_mse == np.inf
    assert res.extras["expansions"] == []
    assert not res.control_mask.any()
    assert np.isnan(did_effect(panel, disc, control=res).tau)
