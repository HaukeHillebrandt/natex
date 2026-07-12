import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import Discovery, lord3_scan
from natex.validate.density import density_test
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
    members yields a NaN placebo p (``Dataset`` drops NaN scan rows but not
    inf; the HC1 regression then returns se = NaN, so t and p are NaN); that
    NaN must stay NaN under Holm and must not inflate the multiplier for the
    finite p-values."""
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", px=3, pz=2, rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    d = res.discoveries[0]
    df = ds.df.copy()
    df["withinf"] = rng.normal(size=len(df))
    df.loc[int(d.members[0]), "withinf"] = np.inf  # one non-finite member value
    spec = DatasetSpec(
        treatment=ds.spec.treatment,
        outcome=ds.spec.outcome,
        forcing=list(ds.spec.forcing),
        covariates=[*ds.spec.covariates, "withinf"],
    )
    rep = placebo_tests(Dataset(df, spec), d)
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
    fails loudly (passed = False) and reports NaN, never 0.0."""
    rng = np.random.default_rng(7)
    n = 80
    z = rng.normal(size=n)
    center = int(np.argsort(z)[n // 2])
    c = rng.normal(size=n)
    c[0] = np.inf  # the ONLY non-forcing covariate has a non-finite member value
    df = pd.DataFrame({"z": z, "T": rng.binomial(1, 0.5, size=n).astype(float), "c": c})
    spec = DatasetSpec(treatment="T", outcome=None, forcing=["z"], covariates=["z", "c"])
    ds = Dataset(df, spec)
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


def test_density_smoke():
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(5))
    rep = density_test(ds, res.discoveries[0])
    assert 0.0 <= rep.p_value <= 1.0
    s = signed_distance(ds, res.discoveries[0])
    assert s.shape == (res.discoveries[0].members.size,)
    assert np.isfinite(s).all()
