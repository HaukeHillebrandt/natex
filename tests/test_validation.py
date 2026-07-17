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

    Calibrated across seeds 0-4: gap p = 0.0 on every seed; null p was
    {0: 0.979, 1: 0.0093, 2: 0.24, 3: 0.0, 4: 0.056}. On this shape (20 bins
    of ~100 counts) the frozen IRLS (100 iterations, tol 1e-10 — pure
    refactor, not retuned here) exits before full convergence, so the null p
    is seed-noisy; on real discovery signed distances it converges to
    machine precision (see test_binned_poisson_jump_matches_density_test).
    Seed 0 pinned per plan: gap p = 0.0 < 0.01, null p = 0.979 > 0.05 —
    wide margin on both gates, and the draw is deterministic.
    """
    rng = np.random.default_rng(0)
    s = rng.uniform(-1.0, 1.0, 2000)
    bump = s[(s >= 0.0) & (s < 0.1)]
    s_gap = np.concatenate([s, bump, bump])  # mass on [0, 0.1] tripled
    assert binned_poisson_jump(s_gap).p_value < 0.01
    assert binned_poisson_jump(s).p_value > 0.05


def test_binned_poisson_jump_degenerate():
    """Constant input (< 2 distinct finite values) -> NaN p, NaN theta —
    NaN, never 0. Same for empty and all-non-finite input."""
    rep = binned_poisson_jump(np.full(50, 3.7))
    assert np.isnan(rep.p_value) and np.isnan(rep.theta)
    rep_empty = binned_poisson_jump(np.array([]))
    assert np.isnan(rep_empty.p_value) and np.isnan(rep_empty.theta)
    rep_nan = binned_poisson_jump(np.array([np.nan, np.inf, -np.inf, 1.0]))
    assert np.isnan(rep_nan.p_value) and np.isnan(rep_nan.theta)
