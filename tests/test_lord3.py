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


def test_issue_1_one_class_error_names_row_loss_offenders():
    """Issue #1: one-sided covariate missingness listwise-deletes every treated
    row; the one-class diagnostic must name the offending column with its
    attributable loss, not leave the user to guess which covariate did it."""
    rng = np.random.default_rng(0)
    n = 60
    T = np.r_[np.zeros(30), np.ones(30)]
    df = pd.DataFrame(
        {
            "T": T,
            "z": rng.normal(size=n),
            "c": np.where(T == 1, np.nan, 1.0),
            "y": rng.normal(size=n),
        }
    )
    ds = Dataset(
        df, DatasetSpec(treatment="T", outcome="y", forcing=["z"], covariates=["z", "c"])
    )
    with pytest.raises(ValueError, match=r"single class[\s\S]*c: 30/60 rows"):
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


def test_issue_9_zero_residual_variance_raises_diagnostic_error():
    """Issue #9: a constant continuous treatment makes the Normal background
    fit exact -- residuals are identically 0, the data-scaled variance floor
    is 0, weights are inf, every LLR is NaN, and NaN won argmax and poisoned
    the randomization p-value (NaN >= NaN is False -> p = 1/(Q+1)). The scan
    must instead fail loudly with a diagnostic ValueError, which discover()
    isolates as status="failed"."""
    n = 20
    df = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n), "T": np.full(n, 2.0)})
    ds = Dataset(df, DatasetSpec(treatment="T", outcome=None, forcing=["x"], covariates=["x"]))
    with pytest.raises(ValueError, match="zero residual variance"):
        lord3_scan(ds, k=8, rng=np.random.default_rng(0))
    # NOTE: an exactly-linear treatment is the same degeneracy in exact
    # arithmetic, but lstsq leaves ~1e-16 float residuals (gv > 0, weights
    # finite, no NaN), so the gv <= 0 guard deliberately does not fire there.


def test_issue_9_local_residual_variance_rejects_zero_residuals():
    """Issue #9 unit level: identically-zero residuals make the variance floor
    0 and every precision weight inf; local_residual_variance must raise a
    diagnostic error rather than return zeros."""
    from natex.scan.neighborhoods import knn_indices, local_residual_variance

    z = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    idx = knn_indices(z, 4)
    with pytest.raises(ValueError, match="zero residual variance"):
        local_residual_variance(np.zeros(12), idx)


# ---------------------------------------------------------------------------
# issue #40: Discovery's indexing contract must be documented at the source


def test_issue_40_discovery_docstring_states_indexing_contract():
    """Issue #40: ``group1`` is a boolean mask ALIGNED WITH ``members``
    (length k), not global row indices — semantics that previously lived only
    in internal planning docs. The contract must be readable on the dataclass
    itself, correct idiom included."""
    from natex.rdd.lord3 import Discovery

    doc = Discovery.__doc__
    assert doc is not None
    flat = " ".join(doc.split())
    assert "members[d.group1]" in flat  # the correct global-indexing idiom
    assert "global" in flat.lower()  # members = global dataset row indices
    assert "mask" in flat.lower()  # group1 = boolean mask over members


def test_issue_40_readme_python_api_shows_the_indexing_idiom():
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "top.members[top.group1]" in readme


def test_issue_40_documented_idiom_holds_on_a_real_scan():
    """The documented semantics, executed: members are global int row indices
    (length k), group1 a length-k bool over them (center on the group-1 side),
    and ``d.members[d.group1]`` / ``d.members[~d.group1]`` partition the
    neighborhood into dataset-indexable sides. Naive ``X[d.group1]`` is the
    dogfooded IndexError."""
    rng = np.random.default_rng(4)
    ds, _ = make_synthetic(n=400, zeta=3.0, kind="binary", rng=rng)
    d = lord3_scan(ds, k=50, rng=np.random.default_rng(5)).discoveries[0]
    assert d.members.dtype.kind == "i" and d.members.shape == (50,)
    assert d.group1.dtype == bool and d.group1.shape == (50,)
    side1 = d.members[d.group1]
    side0 = d.members[~d.group1]
    assert 0 < side1.size < 50 and side0.size == 50 - side1.size
    assert d.center_index in side1  # center always in group 1 (audit item 23)
    assert len(ds.df.iloc[side1]) + len(ds.df.iloc[side0]) == 50
    with pytest.raises(IndexError):
        ds.Z_std[d.group1]  # the naive misuse: mask applied to the full array
