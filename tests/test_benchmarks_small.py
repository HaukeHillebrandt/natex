"""CI-small slices of the KDD ch.5 synthetic benchmark curves (phase-2 task 7).

All calls are seeded (deterministic), so thresholds are hard assertions, not
flaky statistics. Thresholds were chosen with wide margins; if one fails,
suspect the implementation before widening it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from natex.benchmarks import label_noise_curve, nig_power_curve

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_nig_curve.py"

REQUIRED_COLUMNS = [
    "zeta",
    "degree",
    "model",
    "kind",
    "nig_mean",
    "nig_se",
    "power",
    "p_mean",
    "tau_2sls_median",
    "tau_wald_median",
    "n_experiments",
]


def test_curve_schema_and_monotonicity():
    df = nig_power_curve(
        "real", zetas=(0.0, 4.0), n_experiments=3, n=600, k=40, degrees=(1,), Q=19, seed=0
    )
    assert len(df) == 2
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    row0 = df[df["zeta"] == 0.0].iloc[0]
    row4 = df[df["zeta"] == 4.0].iloc[0]
    assert row4["nig_mean"] >= row0["nig_mean"] + 0.2
    assert row4["power"] >= 2 / 3
    assert row0["power"] <= 1 / 3
    assert row0["p_mean"] > 0.25


def test_tau_recovery_strong_signal():
    df = nig_power_curve(
        "real", zetas=(4.0,), n_experiments=3, n=1200, k=50, tau=5.0, Q=19, seed=1
    )
    assert abs(df.iloc[0]["tau_2sls_median"] - 5.0) < 1.5


def test_bernoulli_dominates_normal_on_binary():
    df = nig_power_curve(
        "binary",
        zetas=(4.0,),
        n_experiments=3,
        n=900,
        k=40,
        models=("normal", "bernoulli"),
        Q=19,
        seed=2,
    )
    assert len(df) == 6  # default degrees (1, 2, 4) x 2 models
    nig = df.groupby("model")["nig_mean"].mean()
    # Qualitative Fig-7 direction with slack for the tiny CI sample.
    assert nig["bernoulli"] >= nig["normal"] - 0.05


def test_label_noise_monotone():
    df = label_noise_curve(rhos=(0.6, 1.0), n_experiments=3, n=800, k=40, seed=3)
    assert list(df.columns) == ["rho", "nig_mean", "nig_se", "n_experiments"]
    nig = df.set_index("rho")["nig_mean"]
    assert nig[1.0] >= nig[0.6] + 0.1
    assert nig[1.0] > 0.5
    assert (df["n_experiments"] == 3).all()


def test_determinism():
    kwargs = dict(zetas=(2.0,), n_experiments=2, n=400, k=30, degrees=(1,), Q=9, seed=5)
    a = nig_power_curve("real", **kwargs)
    b = nig_power_curve("real", **kwargs)
    pd.testing.assert_frame_equal(a, b)
    la = label_noise_curve(rhos=(0.8,), n_experiments=2, n=400, k=30, seed=6)
    lb = label_noise_curve(rhos=(0.8,), n_experiments=2, n=400, k=30, seed=6)
    pd.testing.assert_frame_equal(la, lb)
    assert np.isfinite(a["nig_mean"]).all()


def test_script_writes_csvs(tmp_path):
    """Smoke test: the benchmarks script writes CSVs; plots skip gracefully
    when matplotlib is missing (it is an optional extra, absent in default CI)."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--kind", "real",
        "--zetas", "0", "2",
        "--degrees", "1",
        "--n-experiments", "1",
        "--n", "300",
        "--k", "30",
        "--Q", "9",
        "--label-noise",
        "--rhos", "0.9",
        "--noise-experiments", "1",
        "--noise-n", "300",
        "--noise-k", "30",
        "--out", str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    curve = pd.read_csv(tmp_path / "nig_curve_real.csv")
    assert len(curve) == 2
    for col in REQUIRED_COLUMNS:
        assert col in curve.columns
    noise = pd.read_csv(tmp_path / "label_noise.csv")
    assert list(noise.columns) == ["rho", "nig_mean", "nig_se", "n_experiments"]
