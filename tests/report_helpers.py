"""Shared bundle factories for report tests (plain helper module, NOT a conftest).

Every factory is seeded end to end (one ``np.random.default_rng(seed)`` per
stochastic call chain) so bundle contents are deterministic across the report
test modules that import them.
"""

from __future__ import annotations

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


def make_did_bundle(
    tmp_path, *, seed: int = 0
) -> tuple[ResultsBundle, DiscoverReport, Dataset]:
    """Seeded synthetic did discover run wrapped and saved as a bundle."""
    ds, _ = make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(seed))
    report = discover(ds, rng=np.random.default_rng(seed), budget=SMALL)
    bundle = ResultsBundle.from_discover(report, tmp_path, dataset=ds, seed=seed)
    bundle.save()
    return bundle, report, ds
