import numpy as np
import pytest

from natex.data.synthetic import make_synthetic
from natex.rdd.lord3 import lord3_scan
from natex.validate.randomization import randomization_test


def _no_discontinuity_dataset(n, rng, kind="binary"):
    ds, _ = make_synthetic(n=n, zeta=0.0, kind=kind, rng=rng)
    return ds


def test_power_on_strong_signal():
    rng = np.random.default_rng(0)
    ds, _ = make_synthetic(n=800, zeta=5.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=30, rng=np.random.default_rng(1))
    rep = randomization_test(ds, res, Q=19, rng=np.random.default_rng(2), scan_kwargs={"k": 30})
    assert rep.p_value <= 0.05  # 1/(19+1)


def test_null_p_value_not_degenerate():
    rng = np.random.default_rng(3)
    ds = _no_discontinuity_dataset(600, rng, kind="real")
    res = lord3_scan(ds, k=30, rng=np.random.default_rng(4))
    rep = randomization_test(ds, res, Q=19, rng=np.random.default_rng(5), scan_kwargs={"k": 30})
    assert rep.p_value > 0.05  # null data should not (usually) reject at the floor


def test_bernoulli_replicas_are_bernoulli():
    from natex.validate.randomization import _draw_null_treatment

    p_hat = np.full(20000, 0.1)
    t_star = _draw_null_treatment("bernoulli", p_hat, None, np.random.default_rng(6))
    assert set(np.unique(t_star)) <= {0.0, 1.0}
    assert abs(t_star.mean() - 0.1) < 0.01  # NOT ~0.176 like the legacy generator


def test_issue_9_nonfinite_observed_llr_rejected():
    """Issue #9 defense in depth: a non-finite max LLR must never be ranked
    (NaN >= NaN is False, so a NaN observed statistic silently yielded the
    minimum attainable p-value 1/(Q+1))."""
    rng = np.random.default_rng(3)
    ds, _ = make_synthetic(n=200, zeta=0.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=20, rng=np.random.default_rng(4))
    res.discoveries[0].llr = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        randomization_test(ds, res, Q=3, rng=np.random.default_rng(5), scan_kwargs={"k": 20})


def test_issue_25_q_below_one_rejected():
    """Issue #25: Q=0 fabricated a vacuous p=1.0 from zero calibration draws and
    Q=-1 crashed with a raw numpy error; both must fail loudly, mirroring the
    panel contract (validate/panel.py)."""
    rng = np.random.default_rng(3)
    ds, _ = make_synthetic(n=200, zeta=0.0, kind="real", rng=rng)
    res = lord3_scan(ds, k=20, rng=np.random.default_rng(4))
    for bad_q in (0, -1):
        with pytest.raises(ValueError, match="Q must be >= 1"):
            randomization_test(
                ds, res, Q=bad_q, rng=np.random.default_rng(5), scan_kwargs={"k": 20}
            )


def test_issue_25_empty_discoveries_rejected():
    """Issue #25: an empty scan result raised a raw IndexError; there is nothing
    to calibrate, so fail loudly like the panel counterpart."""
    from natex.rdd.lord3 import LoRD3Result

    rng = np.random.default_rng(3)
    ds, _ = make_synthetic(n=200, zeta=0.0, kind="real", rng=rng)
    empty = LoRD3Result(discoveries=[], model="normal", k=20)
    with pytest.raises(ValueError, match="no discoveries"):
        randomization_test(ds, empty, Q=3, rng=np.random.default_rng(5), scan_kwargs={"k": 20})
