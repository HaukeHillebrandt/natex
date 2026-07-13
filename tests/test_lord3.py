import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.rdd.metrics import normalized_information_gain


def test_single_class_treatment_raises_diagnostic_error():
    """Dogfood regression (Fitbit run): listwise deletion can leave a one-class
    treatment; the scan must name the treatment column and the row count, not
    surface sklearn's raw 'needs samples of at least 2 classes' error."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {"T": np.ones(n), "z": rng.normal(size=n), "y": rng.normal(size=n)}
    )
    ds = Dataset(df, DatasetSpec(treatment="T", outcome="y", forcing=["z"], covariates=["z"]))
    with pytest.raises(ValueError, match=r"treatment 'T' has a single class"):
        lord3_scan(ds, k=10, rng=np.random.default_rng(0))


def test_scan_finds_planted_boundary_real_T():
    rng = np.random.default_rng(0)
    ds, D = make_synthetic(n=1500, zeta=4.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(1))
    top = res.top(10)
    # at least one of the top-10 centers sits near the true boundary
    boundary_hit = False
    for d in top:
        raw = ds.Z[d.center_index]
        if np.any(np.abs(raw - 0.5) < 0.15):
            boundary_hit = True
    assert boundary_hit
    # and its split aligns with the truth reasonably well
    nig = normalized_information_gain(D, top[0].members, top[0].group1)
    assert nig > 0.2


def test_scan_binary_model_autoselects():
    rng = np.random.default_rng(2)
    ds, _ = make_synthetic(n=1200, zeta=3.0, kind="binary", rng=rng)
    res = lord3_scan(ds, k=40, rng=np.random.default_rng(3))
    assert res.model == "bernoulli"
    assert res.discoveries[0].llr > 0


def test_outcome_never_read():
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=400, rng=rng)
    ds.df["y"] = np.nan  # poison the outcome; scan must not care
    res = lord3_scan(ds, k=20, rng=np.random.default_rng(5))
    assert np.isfinite(res.discoveries[0].llr)


def test_determinism():
    ds, _ = make_synthetic(n=400, rng=np.random.default_rng(6))
    a = lord3_scan(ds, k=20, rng=np.random.default_rng(7))
    b = lord3_scan(ds, k=20, rng=np.random.default_rng(7))
    assert a.discoveries[0].llr == b.discoveries[0].llr
    assert a.discoveries[0].center_index == b.discoveries[0].center_index


def test_issue_5_no_iprint_optimize_warning_from_logistic_fit():
    """Issue #5: sklearn <= 1.6 passes the removed 'iprint' option to
    scipy >= 1.18's L-BFGS-B, emitting one OptimizeWarning per logistic fit
    (one per replica in the randomization test -> hundreds of stderr lines).
    Exactly that upstream, data-independent warning must be suppressed at the
    fit site; everything else (convergence, separation) stays visible."""
    import warnings

    from natex.rdd.lord3 import fit_treatment_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    T = (rng.uniform(size=60) < 0.5).astype(float)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit_treatment_model(X, T, "bernoulli", 1)
    assert not [w for w in rec if "iprint" in str(w.message)]
