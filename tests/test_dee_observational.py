"""Tests for dee/observational.py: ObservationalEstimator protocol, sklearn
T-learner default, and leave-experiment-out cross-fitting (audit 9).

Regression targets from docs/math_audit_final.md audit 9: the bias observation
at each VKNN center must come from a model whose training rows exclude that
experiment's members (poisoned-member leak test); NaN never 0.0 on failure.

Calibration (seeds 0-4, 2026-07-11): recovery |mean-2| <= 0.026 (bound 0.3);
confounded grid mean 6.70-6.85 (bound > 3); cross-fit max |out-2| <= 0.284
(bound 0.75); leak delta_crossfit = 0.0 exactly (bound < 5, the excluded
training set is bitwise identical) vs delta_fullfit 816-935 (bound > 50).
"""

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.dee.observational import (
    ObservationalEstimator,
    TLearner,
    _assign_folds,
    default_factory,
    experiment_crossfit_cate,
)
from natex.dee.vknn import QuasiExperiment, VKNNResult


def make_dataset(Z, T, y, outcome="y"):
    Z = np.asarray(Z, dtype=float)
    cols = [f"z{j}" for j in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df["T"] = np.asarray(T, dtype=float)
    if outcome is not None:
        df[outcome] = np.asarray(y, dtype=float)
    spec = DatasetSpec(treatment="T", outcome=outcome, forcing=cols, covariates=cols)
    return Dataset(df, spec)


def make_experiments(dataset, centers, size=60):
    """Disjoint QuasiExperiments: `size` nearest not-yet-used rows per raw-unit center."""
    Zs = dataset.Z_std
    Cs = dataset.standardize(np.asarray(centers, dtype=float))
    used = np.zeros(Zs.shape[0], dtype=bool)
    exps = []
    for c in Cs:
        d2 = ((Zs - c) ** 2).sum(axis=1)
        d2[used] = np.inf
        members = np.sort(np.argsort(d2)[:size])
        used[members] = True
        centroid = Zs[members].mean(axis=0)
        exps.append(
            QuasiExperiment(
                center_index=int(members[0]),
                members=members,
                group1=Zs[members, 0] >= c[0],
                normal=np.array([1.0, 0.0]),
                llr=1.0,
                centroid=centroid,
                projected_center=centroid,
            )
        )
    return VKNNResult(
        experiments=exps,
        accepted=np.arange(len(exps)),
        rejected=np.array([], dtype=int),
        k_prime=size,
        t_side=1,
    )


CENTERS = [(0.2, 0.2), (0.2, 0.8), (0.5, 0.5), (0.8, 0.2), (0.8, 0.8), (0.65, 0.35)]


def crossfit_fixture(seed=0, n=1500):
    """U[0,1]^2 forcing, Bern(0.5) treatment, y = sin(3 z0) + 2 T + noise; 6 experiments."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0.0, 1.0, size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    y = np.sin(3.0 * Z[:, 0]) + 2.0 * T + 0.3 * rng.normal(size=n)
    ds = make_dataset(Z, T, y)
    result = make_experiments(ds, CENTERS)
    return Z, T, y, ds, result


# ---------------------------------------------------------------- protocol


def test_protocol_conformance():
    assert isinstance(TLearner(seed=0), ObservationalEstimator)

    class Duck:
        def fit(self, X, T, y):
            return self

        def predict_cate(self, Xq):
            return np.zeros(len(Xq))

    class NotAnEstimator:
        def fit(self, X, T, y):
            return self

    assert isinstance(Duck(), ObservationalEstimator)
    assert not isinstance(NotAnEstimator(), ObservationalEstimator)


# ---------------------------------------------------------------- T-learner


def test_tlearner_recovers_constant_effect():
    rng = np.random.default_rng(3)
    n = 2000
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    T = rng.binomial(1, 0.5, size=n).astype(float)
    y = np.sin(3.0 * X[:, 0]) + 2.0 * T + 0.3 * rng.normal(size=n)
    g = np.linspace(0.05, 0.95, 20)
    grid = np.array([[a, b] for a in g for b in g])
    model = TLearner(seed=0).fit(X, T, y)
    cate = model.predict_cate(grid)
    assert cate.shape == (400,)
    assert abs(float(np.mean(cate)) - 2.0) < 0.3


def test_tlearner_determinism():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(400, 2))
    T = rng.binomial(1, 0.5, size=400).astype(float)
    y = 2.0 * T + rng.normal(size=400)
    grid = rng.uniform(size=(50, 2))
    a = TLearner(seed=11).fit(X, T, y).predict_cate(grid)
    b = TLearner(seed=11).fit(X, T, y).predict_cate(grid)
    assert np.array_equal(a, b)


def test_tlearner_requires_binary_treatment():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 2))
    T = rng.integers(0, 3, size=200).astype(float)  # {0, 1, 2}: not binary
    y = rng.normal(size=200)
    with pytest.raises(ValueError, match=r"econml"):
        TLearner(seed=0).fit(X, T, y)


def test_tlearner_underdetermined_arm_gives_nan():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(100, 2))
    T = np.zeros(100)
    T[:5] = 1.0  # 5 treated < min_treated=20
    y = rng.normal(size=100)
    cate = TLearner(seed=0).fit(X, T, y).predict_cate(rng.uniform(size=(7, 2)))
    assert cate.shape == (7,)
    assert np.all(np.isnan(cate))  # NaN, never 0.0


def test_tlearner_predict_before_fit_raises():
    with pytest.raises(ValueError, match=r"fit"):
        TLearner(seed=0).predict_cate(np.zeros((3, 2)))


def test_confounded_dgp_shows_bias():
    # y = 2T + 3u with u unobserved and T = 1{u > 0}: the conditional-on-X
    # contrast is 2 + 3*E[u|u>0]*2 ~ 6.8, so the T-learner MUST be biased
    # upward (> 3) -- otherwise DEE would have nothing to fix.
    rng = np.random.default_rng(4)
    n = 4000
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    u = rng.normal(size=n)
    T = (u > 0.0).astype(float)
    y = 2.0 * T + 3.0 * u + 0.3 * rng.normal(size=n)
    g = np.linspace(0.05, 0.95, 20)
    grid = np.array([[a, b] for a in g for b in g])
    cate = TLearner(seed=0).fit(X, T, y).predict_cate(grid)
    assert float(np.mean(cate)) > 3.0


# ---------------------------------------------------------------- factory


def test_default_factory_seeds_independently_and_deterministically():
    fac = default_factory(np.random.default_rng(9))
    a, b = fac(), fac()
    assert isinstance(a, TLearner) and isinstance(b, TLearner)
    assert a.seed != b.seed
    fac2 = default_factory(np.random.default_rng(9))
    assert (fac2().seed, fac2().seed) == (a.seed, b.seed)


def test_default_factory_requires_rng():
    with pytest.raises(ValueError, match="Generator"):
        default_factory(None)


# ---------------------------------------------------------------- cross-fitting


def test_fold_assignment_covers_every_experiment_exactly_once():
    for u in [1, 2, 5, 7, 12]:
        fold = _assign_folds(u, 5, np.random.default_rng(u))
        f = min(5, u)
        assert fold.shape == (u,)
        assert fold.min() >= 0 and fold.max() == f - 1
        assert np.all(np.bincount(fold, minlength=f) > 0)  # every fold non-empty
    with pytest.raises(ValueError):
        _assign_folds(3, 0, np.random.default_rng(0))


def test_crossfit_shape_determinism_and_recovery():
    _, _, _, ds, result = crossfit_fixture(seed=0)

    def run():
        return experiment_crossfit_cate(
            ds, result, default_factory(np.random.default_rng(1)), np.random.default_rng(2)
        )

    out1, out2 = run(), run()
    assert out1.shape == (6,)
    assert np.array_equal(out1, out2)  # bitwise-identical under the same rng
    assert np.all(np.isfinite(out1))
    # true CATE is 2 everywhere; centroid predictions should be close
    assert np.max(np.abs(out1 - 2.0)) < 0.75


def test_crossfit_empty_result():
    _, _, _, ds, _ = crossfit_fixture(seed=0, n=200)
    empty = VKNNResult(
        experiments=[],
        accepted=np.array([], dtype=int),
        rejected=np.array([], dtype=int),
        k_prime=1,
        t_side=1,
    )
    out = experiment_crossfit_cate(
        ds, empty, default_factory(np.random.default_rng(0)), np.random.default_rng(0)
    )
    assert out.shape == (0,)


def test_poisoned_member_leak_regression():
    # Audit 9: poison one experiment's TREATED members (arm-symmetric poison
    # would cancel in the mu1 - mu0 contrast) and check the cross-fitted
    # prediction at that centroid is unmoved while a full-fit prediction jumps.
    Z, T, y, ds_clean, result = crossfit_fixture(seed=0)
    j = 2
    members = result.experiments[j].members
    y_poison = y.copy()
    treated_members = members[T[members] == 1.0]
    assert treated_members.size >= 10
    y_poison[treated_members] += 1000.0
    ds_poison = make_dataset(Z, T, y_poison)

    def crossfit(ds):
        return experiment_crossfit_cate(
            ds, result, default_factory(np.random.default_rng(1)), np.random.default_rng(2)
        )

    delta_cf = abs(crossfit(ds_poison)[j] - crossfit(ds_clean)[j])
    assert delta_cf < 5.0

    center = result.experiments[j].projected_center[None, :]

    def full_fit(ds):
        model = default_factory(np.random.default_rng(1))()
        return float(model.fit(ds.Z_std, ds.T, ds.y).predict_cate(center)[0])

    delta_full = abs(full_fit(ds_poison) - full_fit(ds_clean))
    assert delta_full > 50.0


def test_all_control_dataset_yields_nan():
    rng = np.random.default_rng(6)
    n = 800
    Z = rng.uniform(0.0, 1.0, size=(n, 2))
    T = np.zeros(n)
    y = np.sin(3.0 * Z[:, 0]) + 0.3 * rng.normal(size=n)
    ds = make_dataset(Z, T, y)
    result = make_experiments(ds, CENTERS[:3])
    out = experiment_crossfit_cate(
        ds, result, default_factory(np.random.default_rng(0)), np.random.default_rng(0)
    )
    assert out.shape == (3,)
    assert np.all(np.isnan(out))  # NaN, never 0.0
    assert not np.any(out == 0.0)


def test_locally_untreated_experiment_still_finite():
    # An experiment whose members are ALL control still gets a finite
    # prediction: the fold model is global (treated support elsewhere).
    rng = np.random.default_rng(7)
    n = 1500
    Z = rng.uniform(0.0, 1.0, size=(n, 2))
    T = ((Z[:, 0] > 0.4) & (rng.uniform(size=n) < 0.5)).astype(float)
    y = 2.0 * T + Z[:, 0] + 0.3 * rng.normal(size=n)
    ds = make_dataset(Z, T, y)
    result = make_experiments(ds, [(0.15, 0.5), (0.7, 0.5)])
    assert np.all(T[result.experiments[0].members] == 0.0)
    out = experiment_crossfit_cate(
        ds, result, default_factory(np.random.default_rng(0)), np.random.default_rng(0)
    )
    assert np.all(np.isfinite(out))


def test_crossfit_tolerates_nan_outcomes():
    Z, T, y, _, result = crossfit_fixture(seed=8)
    y = y.copy()
    y[:50] = np.nan  # rows kept by Dataset (outcome never in the dropna set)
    ds = make_dataset(Z, T, y)
    assert ds.n == len(y)
    out = experiment_crossfit_cate(
        ds, result, default_factory(np.random.default_rng(0)), np.random.default_rng(0)
    )
    assert np.all(np.isfinite(out))


def test_crossfit_requires_outcome_and_rng():
    rng = np.random.default_rng(0)
    Z = rng.uniform(size=(300, 2))
    T = rng.binomial(1, 0.5, size=300).astype(float)
    ds_noy = make_dataset(Z, T, None, outcome=None)
    result = make_experiments(ds_noy, [(0.5, 0.5)], size=40)
    with pytest.raises(ValueError, match="outcome"):
        experiment_crossfit_cate(
            ds_noy, result, default_factory(np.random.default_rng(0)), np.random.default_rng(0)
        )
    ds = make_dataset(Z, T, rng.normal(size=300))
    with pytest.raises(ValueError, match="Generator"):
        experiment_crossfit_cate(ds, result, default_factory(np.random.default_rng(0)), None)
