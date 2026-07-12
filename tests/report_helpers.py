"""Shared bundle factories for report tests (plain helper module, NOT a conftest).

Every factory is seeded end to end (one ``np.random.default_rng(seed)`` per
stochastic call chain) so bundle contents are deterministic across the report
test modules that import them.
"""

from __future__ import annotations

import json

import numpy as np

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic
from natex.discover import DiscoverReport, discover
from natex.report.bundle import ResultsBundle

SMALL = {"k": 25, "q": 9}  # every scan here: n <= 400, q = 9, seeded


def make_rdd_bundle(
    tmp_path, *, with_outcome: bool = True, seed: int = 0
) -> tuple[ResultsBundle, DiscoverReport, Dataset]:
    """Seeded synthetic rdd discover run wrapped and saved as a bundle."""
    ds, _ = make_synthetic(n=300, zeta=6.0, kind="binary", rng=np.random.default_rng(seed))
    if not with_outcome:
        df = ds.df.drop(columns=[ds.spec.outcome])
        spec = DatasetSpec(
            treatment=ds.spec.treatment, outcome=None,
            forcing=list(ds.spec.forcing), covariates=list(ds.spec.covariates),
        )
        ds = Dataset(df, spec)
    report = discover(ds, rng=np.random.default_rng(seed), budget=SMALL)
    bundle = ResultsBundle.from_discover(report, tmp_path, dataset=ds, seed=seed)
    bundle.save()
    return bundle, report, ds


# Plain single-scan results.json payloads exactly as the non-plan `natex
# discover` CLI paths write them (no "natex_bundle" marker) — the F-D1 input
# form README section "Reports and papers" promises paper/brief accept.
RDD_SCAN_PAYLOAD = {
    "params": {"k": 40, "q": 49, "seed": 0, "degree": 1, "coarse": False,
               "n_coarse": 2000, "csv": "synth.csv"},
    "scan": {"model": "bernoulli", "p_value": 0.02, "observed_max_llr": 21.16},
    "discoveries": [
        {"center_z": [0.758, 0.513], "llr": 21.16, "normal": [0.94, -0.34],
         "forcing_influence": {"x0": 0.94, "x1": 0.34}},
        {"center_z": [0.71, 0.50], "llr": 19.0, "normal": [0.9, -0.44],
         "forcing_influence": {"x0": 0.9, "x1": 0.44}},
    ],
    "validation": {"placebo_holm": {"x2": 0.7}, "placebo_passed": True,
                   "density_p": 0.559},
    "effects": {
        "2sls": {"tau": 2.047, "se": 0.522, "ci": [1.024, 3.07],
                 "first_stage_t": 8.83, "weak_instrument": False},
        "wald": {"tau": 2.1, "se": 0.6, "ci": [0.9, 3.3],
                 "first_stage_t": 8.83, "weak_instrument": False},
    },
}

DID_SCAN_PAYLOAD = {
    "params": {"design": "did", "q": 49, "seed": 0, "degree": 1, "time": "t",
               "unit": "state", "bins": 4, "windows": None, "restarts": 8,
               "method": "single_delta", "model": "auto", "csv": "panel.csv"},
    "did": {
        "scan": {"model": "normal", "method": "single_delta", "p_value": 0.04,
                 "observed_max_llr": 9.3, "null_kind": "ar1_unit"},
        "discoveries": [
            {"subset_values": {"x0": [1, 2]}, "t0": 15.0, "window": 8.0, "llr": 9.3}
        ],
        "validation": {"composition_p": 0.4, "composition_passed": True,
                       "anticipation_shifts": [1, 2],
                       "anticipation_p_holm": [0.3, 0.6],
                       "anticipation_passed": True},
        "effects": {"dd": {"tau": -0.2, "se": 0.1, "p": 0.03,
                           "pre_mse": 0.01, "dose": None}},
        "searched": {"windows": [8.0], "restarts": 8, "method": "single_delta",
                     "model": "normal", "dims": ["x0"], "bin_counts": {"x0": 4}},
    },
}


def make_scan_payload_bundle(tmp_path, *, design: str = "rdd") -> tuple[ResultsBundle, dict]:
    """Write a plain single-scan results.json and load it as a bundle (F-D1)."""
    payload = RDD_SCAN_PAYLOAD if design == "rdd" else DID_SCAN_PAYLOAD
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return ResultsBundle.load(tmp_path), payload


def make_did_bundle(
    tmp_path, *, seed: int = 0
) -> tuple[ResultsBundle, DiscoverReport, Dataset]:
    """Seeded synthetic did discover run wrapped and saved as a bundle."""
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(seed))
    report = discover(ds, rng=np.random.default_rng(seed), budget=SMALL)
    bundle = ResultsBundle.from_discover(report, tmp_path, dataset=ds, seed=seed)
    bundle.save()
    return bundle, report, ds
