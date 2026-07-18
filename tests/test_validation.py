import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import Discovery, lord3_scan
from natex.validate.density import binned_poisson_jump, density_test
from natex.validate.honest import honest_split
from natex.validate.placebo import _holm, hc1_ols, placebo_tests, signed_distance


def test_honest_split_disjoint_and_deterministic():
    a1, b1 = honest_split(100, rng=np.random.default_rng(0))
    a2, b2 = honest_split(100, rng=np.random.default_rng(0))
    assert set(a1) & set(b1) == set()
    assert len(a1) + len(b1) == 100
    np.testing.assert_array_equal(a1, a2)


def test_hc1_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + rng.normal(size=500)
    X = np.c_[np.ones(500), x]
    beta, se = hc1_ols(X, y)
    assert abs(beta[1] - 2.0) < 0.2 and abs(beta[2] + 3.0) < 0.2
    assert np.all(se > 0)


def test_placebo_passes_on_clean_synthetic():
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", px=3, pz=2, rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    rep = placebo_tests(ds, res.discoveries[0])
    # x2 (non-forcing covariate) is smooth through the boundary -> should pass
    assert rep.passed


def test_placebo_holm_exact_values_with_nan():
    """F-A1 (audit item 3): NaN p-values are excluded from the Holm family and
    preserved as NaN — never a fabricated finite adjusted p, never 0.0, and
    never counted in the multiplier m."""
    out = _holm({"a": 0.02, "b": float("nan"), "c": 0.5})
    assert out["a"] == 0.04  # m = 2 finite entries, rank 0 -> 2 * 0.02
    assert out["c"] == 0.5  # max(0.04, 1 * 0.5)
    assert np.isnan(out["b"])  # excluded and preserved
    # NaN sorting first must not zero anything (the {b: NaN, a: .02} case)
    out2 = _holm({"b": float("nan"), "a": 0.02})
    assert out2["a"] == 0.02
    assert np.isnan(out2["b"])


def test_placebo_holm_nan_aware_regression():
    """F-A1: a covariate with a non-finite value among the neighborhood
    members yields a NaN placebo p (the HC1 regression returns se = NaN, so t
    and p are NaN); that NaN must stay NaN under Holm and must not inflate the
    multiplier for the finite p-values. Dataset now drops non-finite scan rows
    at construction (issue #20), so the inf is planted post-construction —
    standing in for any degenerate regression that yields a NaN p."""
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", px=3, pz=2, rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    d = res.discoveries[0]
    df = ds.df.copy()
    df["withinf"] = rng.normal(size=len(df))
    spec = DatasetSpec(
        treatment=ds.spec.treatment,
        outcome=ds.spec.outcome,
        forcing=list(ds.spec.forcing),
        covariates=[*ds.spec.covariates, "withinf"],
    )
    ds2 = Dataset(df, spec)
    assert ds2.n == ds.n  # no drops: member indices from the scan stay valid
    ds2.df.loc[int(d.members[0]), "withinf"] = np.inf  # one non-finite member value
    rep = placebo_tests(ds2, d)
    assert np.isnan(rep.p_values["withinf"])
    assert np.isnan(rep.p_holm["withinf"])  # legacy code fabricated a finite value
    finite = {k: v for k, v in rep.p_values.items() if not np.isnan(v)}
    assert finite, "clean synthetic must keep testable covariates"
    m = len(finite)  # the Holm multiplier counts ONLY the finite p-values
    running = 0.0
    for rank, (name, p) in enumerate(sorted(finite.items(), key=lambda kv: kv[1])):
        running = max(running, (m - rank) * p)
        assert rep.p_holm[name] == min(running, 1.0)
    assert rep.passed  # NaN excluded from the decision; battery not vacuous


def test_placebo_all_nan_fails_loud():
    """F-A1 corner: when EVERY testable covariate yields a NaN p, the battery
    fails loudly (passed = False) and reports NaN, never 0.0. Dataset now
    drops non-finite scan rows at construction (issue #20), so the inf is
    planted post-construction to force the NaN-p path."""
    rng = np.random.default_rng(7)
    n = 80
    z = rng.normal(size=n)
    center = int(np.argsort(z)[n // 2])
    df = pd.DataFrame(
        {"z": z, "T": rng.binomial(1, 0.5, size=n).astype(float), "c": rng.normal(size=n)}
    )
    spec = DatasetSpec(treatment="T", outcome=None, forcing=["z"], covariates=["z", "c"])
    ds = Dataset(df, spec)
    ds.df.loc[0, "c"] = np.inf  # the ONLY non-forcing covariate has a non-finite member value
    normal = np.array([1.0])
    group1 = ((ds.Z_std - ds.Z_std[center]) @ normal) >= 0
    d = Discovery(
        center_index=center, k=n, llr=1.0, normal=normal,
        members=np.arange(n), group1=group1,
    )
    rep = placebo_tests(ds, d)
    assert np.isnan(rep.p_values["c"])
    assert np.isnan(rep.p_holm["c"])
    assert rep.passed is False


def test_issue_3_row_unique_and_degenerate_dummies_stay_out_of_holm_family():
    """Issue #3: ``covariates="auto"`` sweeps in string date/ID columns whose
    one-hot levels have within-neighborhood support 1; each entered the Holm
    family as a powerless test, silently diluting genuine placebo failures.
    Row-unique non-numeric columns must be excluded before one-hot encoding,
    degenerate 0/1 levels skipped, and both recorded alongside the family
    size m so the battery is auditable."""
    rng = np.random.default_rng(11)
    n = 80
    z = rng.normal(size=n)
    center = int(np.argsort(z)[n // 2])
    members = np.argsort(np.abs(z - z[center]))[:20]
    grp = np.array(["a"] * n, dtype=object)
    grp[int(members[1])] = "b"  # support 1 inside the neighborhood ...
    grp[int(np.setdiff1d(np.arange(n), members)[0])] = "b"  # ... not row-unique
    df = pd.DataFrame({
        "z": z,
        "T": rng.binomial(1, 0.5, size=n).astype(float),
        "c": rng.normal(size=n),
        "grp": grp,
        "date": pd.date_range("2001-01-01", periods=n).strftime("%Y-%m-%d"),
    })
    spec = DatasetSpec(
        treatment="T", outcome=None, forcing=["z"], covariates=["z", "c", "grp", "date"]
    )
    ds = Dataset(df, spec)
    normal = np.array([1.0])
    group1 = ((ds.Z_std[members] - ds.Z_std[center]) @ normal) >= 0
    d = Discovery(
        center_index=center, k=20, llr=1.0, normal=normal, members=members, group1=group1
    )
    rep = placebo_tests(ds, d)
    # the row-unique date column never reaches the family, level by level
    assert not any(name.startswith("date") for name in rep.p_values)
    assert "date" in rep.skipped
    # both grp levels are degenerate inside the neighborhood (minority count 1)
    assert not any(name.startswith("grp") for name in rep.p_values)
    assert any(name.startswith("grp") for name in rep.skipped)
    # only the genuine covariate enters the Holm family, and m says so
    assert set(rep.p_values) == {"c"}
    assert rep.m == 1


def test_density_smoke():
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(5))
    rep = density_test(ds, res.discoveries[0])
    assert 0.0 <= rep.p_value <= 1.0
    s = signed_distance(ds, res.discoveries[0])
    assert s.shape == (res.discoveries[0].members.size,)
    assert np.isfinite(s).all()


def test_binned_poisson_jump_matches_density_test():
    """Pure-refactor proof (phase survey, task 4): density_test must delegate
    to binned_poisson_jump, so on the same signed distances both return the
    IDENTICAL statistic bitwise — same p_value, same theta."""
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(5))
    d = res.discoveries[0]
    rep = density_test(ds, d)
    rep2 = binned_poisson_jump(signed_distance(ds, d))
    assert rep.p_value == rep2.p_value  # bitwise, not approx
    assert rep.theta == rep2.theta


def test_binned_poisson_jump_detects_gap():
    """Tripling the mass on [0, 0.1] of 2000 uniform[-1, 1] draws must yield a
    tiny p; the untouched symmetric sample must not.

    Recalibrated for issue #42 (mean-initialized IRLS, converges in ~4-5
    iterations; the old zero start exited the cap on this shape and the null
    p was seed-noisy): across seeds 0-4 the gap p is < 1e-7 on every seed
    and the null p is in [0.42, 0.89]. Seed 0 pinned per plan: gap
    p ~ 9.3e-10 < 0.01, null p ~ 0.89 > 0.05 — wide margin on both gates,
    and the draw is deterministic.
    """
    rng = np.random.default_rng(0)
    s = rng.uniform(-1.0, 1.0, 2000)
    bump = s[(s >= 0.0) & (s < 0.1)]
    s_gap = np.concatenate([s, bump, bump])  # mass on [0, 0.1] tripled
    assert binned_poisson_jump(s_gap).p_value < 0.01
    assert binned_poisson_jump(s).p_value > 0.05


def test_binned_poisson_jump_degenerate():
    """Constant input (< 2 distinct finite values) -> NaN p, NaN theta, NaN
    se — NaN, never 0. Same for empty and all-non-finite input."""
    rep = binned_poisson_jump(np.full(50, 3.7))
    assert np.isnan(rep.p_value) and np.isnan(rep.theta) and np.isnan(rep.se)
    rep_empty = binned_poisson_jump(np.array([]))
    assert np.isnan(rep_empty.p_value) and np.isnan(rep_empty.theta)
    assert np.isnan(rep_empty.se)
    rep_nan = binned_poisson_jump(np.array([np.nan, np.inf, -np.inf, 1.0]))
    assert np.isnan(rep_nan.p_value) and np.isnan(rep_nan.theta)
    assert np.isnan(rep_nan.se)


def test_binned_poisson_jump_smooth_null_calibrated():
    """Issue #42: with the zero-initialized IRLS the smooth-null fit exited
    at the iteration cap on ~100-count bins and reported a fabricated finite
    p that was seed-noisy (p = 0.0 at seed 3 on a plain uniform!). With the
    mean-initialized IRLS the fit converges in ~4 iterations and no smooth
    null rejects. Calibrated (n=2000 uniform[-1,1], seeds {0..4, 42}):
    p in [0.14, 0.89] — every seed clears the 0.01 gate by >= 14x."""
    for seed in (1, 3, 42):
        rng = np.random.default_rng(seed)
        rep = binned_poisson_jump(rng.uniform(-1.0, 1.0, 2000))
        assert rep.p_value > 0.01, f"seed {seed}: smooth null rejected (p={rep.p_value})"


def test_binned_poisson_jump_nonconvergence_is_nan():
    """Issue #42: a fit that exits the IRLS loop without converging must
    report NaN (p, theta, se) — never a fabricated finite statistic from
    wherever the iteration cap happened to leave beta (NaN-never-0.0)."""
    rng = np.random.default_rng(0)
    s = rng.uniform(-1.0, 1.0, 2000)
    bump = s[(s >= 0.0) & (s < 0.1)]
    rep = binned_poisson_jump(np.concatenate([s, bump, bump]), max_iter=1)
    assert np.isnan(rep.p_value) and np.isnan(rep.theta) and np.isnan(rep.se)


def test_binned_poisson_jump_window():
    """Issue #42: ``window`` restricts the fit to |s| <= window so the GLM
    tests the LOCAL density jump instead of binning the full data range.

    Far-mass invariance: mass entirely outside the window must not move the
    statistic — the report on (base + far mass, window=w) equals the report
    on the base sample alone at the same window, bitwise (issue #43 pins the
    windowed edges to [-w, w], so the comparison holds regardless of the
    realized data range). Localization (seed 3, calibrated): 4000
    uniform[-10,10] plus 300 planted on [-0.4, 0): the windowed fit
    concentrates the statistic (|theta| 2.1 vs 0.42 diluted, p ~1e-35 vs
    ~2e-12). The <2-distinct-values guard applies AFTER subsetting: a window
    that empties the sample yields NaN, never a crash or 0."""
    rng = np.random.default_rng(7)
    base = rng.uniform(-0.5, 0.5, 1500)
    far = rng.uniform(2.0, 3.0, 500)
    rep_base = binned_poisson_jump(base, window=0.5)
    rep_win = binned_poisson_jump(np.concatenate([base, far]), window=0.5)
    assert rep_win.p_value == rep_base.p_value  # bitwise
    assert rep_win.theta == rep_base.theta
    assert rep_win.se == rep_base.se

    rng = np.random.default_rng(3)
    sb = np.concatenate([rng.uniform(-10.0, 10.0, 4000), rng.uniform(-0.4, 0.0, 300)])
    rep_full = binned_poisson_jump(sb)
    rep_local = binned_poisson_jump(sb, window=1.0)
    assert abs(rep_local.theta) > abs(rep_full.theta)
    assert rep_local.p_value < 1e-20

    rep_empty = binned_poisson_jump(np.array([5.0, 6.0, 7.0]), window=1.0)
    assert np.isnan(rep_empty.p_value) and np.isnan(rep_empty.theta)
    assert np.isnan(rep_empty.se)


def test_binned_poisson_jump_pins_bin_edge_at_zero():
    """Issue #43: edges ran min(s) -> max(s) with no regard for the cutoff,
    so with asymmetric support one bin straddled 0 and its observations were
    assigned wholesale to the sign of the bin MID — mixing the two sides,
    attenuating theta, and letting the far tail (via min/max) silently steer
    the side split. Edges are now built per side, pinned at 0, with a
    log-bin-width exposure offset so unequal per-side widths cannot
    masquerade as a jump.

    Exact-grid pin: with window=1.0 the edges are deterministic (10 bins per
    side, width 0.1); atoms at the 20 bin centers with 10 copies/bin left and
    20 copies/bin right are a perfect piecewise-constant fit, so theta must
    equal ln 2 to machine precision (old min->max edges misalign with the
    atoms and miss). Statistical pin (x4 jump, support [-3, 0.6], 6000 left /
    4800 right): calibrated over seeds 0-7 the new error |theta - ln 4| is
    <= 0.036 while the old edges attenuate by 0.115-0.278 on every seed;
    seed 0 pinned (error 0.001 vs 0.226) with gate 0.08."""
    centers = -0.95 + 0.1 * np.arange(20)
    s = np.concatenate([np.repeat(centers[:10], 10), np.repeat(centers[10:], 20)])
    rep = binned_poisson_jump(s, window=1.0)
    assert np.isclose(rep.theta, np.log(2.0), atol=1e-8)

    rng = np.random.default_rng(0)
    s4 = np.concatenate([rng.uniform(-3.0, 0.0, 6000), rng.uniform(0.0, 0.6, 4800)])
    rep4 = binned_poisson_jump(s4)
    assert abs(rep4.theta - np.log(4.0)) < 0.08
    assert rep4.p_value < 1e-100


def test_binned_poisson_jump_one_sided_support_is_nan():
    """Issue #43: with every observation on one side of the cutoff there is
    no jump to test; the old code still fit the GLM (side indicator constant
    across bins) and returned a fabricated finite theta. NaN, never 0."""
    rep = binned_poisson_jump(np.linspace(0.5, 3.0, 50))
    assert np.isnan(rep.p_value) and np.isnan(rep.theta) and np.isnan(rep.se)
    rep_neg = binned_poisson_jump(-np.linspace(0.5, 3.0, 50))
    assert np.isnan(rep_neg.p_value) and np.isnan(rep_neg.theta)


def test_density_test_threads_window():
    """Issue #43: density_test forwards ``window`` to binned_poisson_jump so
    the frozen-geometry path can also test the LOCAL density jump."""
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(5))
    d = res.discoveries[0]
    s = signed_distance(ds, d)
    w = float(np.quantile(np.abs(s), 0.8))
    rep = density_test(ds, d, window=w)
    rep2 = binned_poisson_jump(s, window=w)
    assert rep.p_value == rep2.p_value and rep.theta == rep2.theta


def test_binned_poisson_jump_min_per_side_guard():
    """Issue #44: 2 observations on one side of the cutoff (the Epoch shape,
    19/2 at n=21) still returned a finite theta from <= 2 informative bins of
    a 4-parameter GLM (20/1 returned theta = -175). Below ``min_per_side``
    (default 5) the report must refuse with NaN and say why (n_left, n_right,
    note) — NaN, never 0."""
    rng = np.random.default_rng(0)
    s = np.concatenate([-rng.uniform(0.1, 3.0, 19), rng.uniform(0.0, 3.0, 2)])
    rep = binned_poisson_jump(s)
    assert np.isnan(rep.p_value) and np.isnan(rep.theta) and np.isnan(rep.se)
    assert rep.n_left == 19 and rep.n_right == 2
    assert rep.note is not None and "5" in rep.note

    # 20/1 (the theta = -175 shape) refuses identically
    s2 = np.concatenate([-rng.uniform(0.1, 3.0, 20), rng.uniform(0.0, 3.0, 1)])
    rep2 = binned_poisson_jump(s2)
    assert np.isnan(rep2.theta) and rep2.n_right == 1

    # the boundary runs: 5/5 at the default returns a finite report
    s3 = np.concatenate([-rng.uniform(0.1, 3.0, 5), rng.uniform(0.0, 3.0, 5)])
    rep3 = binned_poisson_jump(s3)
    assert np.isfinite(rep3.p_value) and rep3.n_left == 5 and rep3.n_right == 5
    assert rep3.note is None

    # explicit override: min_per_side=2 lets the 19/2 shape through, noted or not
    rep4 = binned_poisson_jump(s, min_per_side=2)
    assert np.isfinite(rep4.p_value)

    # side convention matches the bin indicator: s >= 0 counts as right, so a
    # right side made only of exact zeros is counted there (and the support
    # then fails to straddle the cutoff -> NaN with the counts surfaced)
    z = np.concatenate([-rng.uniform(0.1, 1.0, 10), np.zeros(6)])
    repz = binned_poisson_jump(z)
    assert repz.n_left == 10 and repz.n_right == 6
    assert np.isnan(repz.p_value)


def test_density_report_carries_wald_se():
    """Issue #41: the Wald SE the GLM already computes must be surfaced as
    ``DensityReport.se`` — reverse-engineering it as theta/isf(p/2) breaks
    when p underflows to exactly 0.0 (isf -> inf) and is 0/0 when theta == 0.

    Benign regime (task-4 seed-0 tripled-mass draw): se is finite, positive,
    and internally consistent with the reported Wald p. Extreme regime
    (right-side mass x100, n=20000): the reported p underflows toward 0 so
    the isf hack degenerates, yet se stays finite and positive."""
    from scipy import stats

    rng = np.random.default_rng(0)
    s = rng.uniform(-1.0, 1.0, 2000)
    bump = s[(s >= 0.0) & (s < 0.1)]
    rep = binned_poisson_jump(np.concatenate([s, bump, bump]))
    assert np.isfinite(rep.se) and rep.se > 0
    assert np.isclose(rep.p_value, 2 * stats.norm.sf(abs(rep.theta / rep.se)))

    rng = np.random.default_rng(1)
    s_extreme = np.concatenate(
        [rng.uniform(-1.0, 0.0, 200), rng.uniform(0.0, 1.0, 20000)]
    )
    rep_x = binned_poisson_jump(s_extreme)
    assert np.isfinite(rep_x.se) and rep_x.se > 0
    assert rep_x.p_value < 1e-8
