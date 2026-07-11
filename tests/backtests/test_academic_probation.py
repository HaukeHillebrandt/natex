"""Backtest: Academic Probation (Lindo, Sanders & Oreopoulos 2010), 44k rows.

Fuzzy RDD at ``dist_from_cut = 0`` (probation assigned below the first-year GPA
cutoff). This file is the scale gate for the phase-2 scan engineering: the full
44,362-row scan runs through ``coarse_to_fine_scan`` with k=75 and
n_coarse=3000 (the paper used k=100; k=75 keeps the per-center partition count
manageable and the whole file inside the ~10-minute budget, so n_coarse stayed
at the planned 3000 — no reduction to 2000 was needed).

Known truths asserted (audit item 20: legacy outputs are NOT parity targets —
we validate against the dataset's documented truth and the paper's qualitative
Table-2/3 results):

- the scan localizes at dist_from_cut = 0 and ranks dist_from_cut #1 of the 4
  forcing variables by mean |normal| component (paper: Bernoulli influence on
  GPA-distance 1.0 +/- 0.0);
- the max-LLR is significant under the fitted-null bootstrap (on a seeded
  subsample; see test_scan_significant_on_subsample for the caveat);
- local 2SLS on GPA_year2 (~13% missing outcomes — exercises the NaN-tolerant
  estimator hardening) gives 0 < tau < 1 with a strong first stage (Lindo:
  +0.233; LoRD3-paper 2SLS: ~0.255; the bracket is generous, sign is the
  claim). Estimation widens the bandwidth relative to detection — see
  test_effect_sign_matches_lindo for the power argument.
"""

import numpy as np
import pytest

from natex.data.spec import Dataset
from natex.estimate.local2sls import local_2sls
from natex.rdd.lord3 import lord3_scan
from natex.scan.coarse import coarse_to_fine_scan
from natex.scan.geometry import build_geometry
from natex.validate.randomization import randomization_test

pytestmark = pytest.mark.backtest

DIST = 0  # index of dist_from_cut in the forcing tuple (registry order)


@pytest.fixture(scope="module")
def ds(load_or_skip):
    return load_or_skip("academic_probation")


@pytest.fixture(scope="module")
def ctf(ds):
    # Shared by the cutoff-recovery and effect tests (one 44k scan, not two).
    return coarse_to_fine_scan(ds, k=75, n_coarse=3000, top_m=20, rng=np.random.default_rng(0))


def test_recovers_probation_cutoff_and_forcing_rank(ds, ctf):
    assert ds.treatment_is_binary  # probation_year1 in {0,1} => Bernoulli model
    assert ctf.result.model == "bernoulli"

    top = ctf.result.top(10)
    assert len(top) == 10
    raw_dist = [float(ds.Z[d.center_index, DIST]) for d in top]
    assert any(abs(v) < 0.1 for v in raw_dist), f"no top-10 center near the cutoff: {raw_dist}"

    # dist_from_cut must rank #1 of the 4 forcing variables by mean |normal|
    # component over the top-10 discoveries (paper: 1.0 +/- 0.0 for Bernoulli).
    mean_abs = np.mean(np.abs([d.normal for d in top]), axis=0)
    assert mean_abs.shape == (4,)
    assert int(np.argmax(mean_abs)) == DIST, f"forcing rank wrong: mean |normal| = {mean_abs}"

    # The scaling machinery actually engaged (spec 6b coverage contract): only
    # a minority of the 44k centers was visited, and the budget is on record.
    assert ctf.frac_centers_scanned < 0.5
    assert ctf.params["n_coarse"] == 3000
    assert ctf.params["top_m"] == 20
    assert ctf.params["k"] == 75


def test_scan_significant_on_subsample(ds):
    """Fitted-null significance of the max-LLR scan on a seeded 8k subsample.

    Randomization on the full 44k rows x Q replicas is out of compute budget;
    the statement made here is CONDITIONALLY valid: significance is asserted
    for the scan distribution of this seeded 8000-row subsample, not for the
    full-sample scan statistic. Q=19 makes p = 0.05 the smallest attainable
    p-value, so the assertion requires the observed max-LLR to beat all 19
    fitted-null replicas.
    """
    pick = np.random.default_rng(0).choice(ds.n, size=8000, replace=False)
    sub = Dataset(ds.df.iloc[pick].reset_index(drop=True), ds.spec)
    assert sub.n == 8000

    # One geometry (kNN + partition cache) shared by the observed scan and all
    # replicas: bit-identical results, large constant-factor savings.
    geometry = build_geometry(sub.Z_std, 75)
    res = lord3_scan(sub, k=75, geometry=geometry)
    rep = randomization_test(
        sub, res, Q=19, rng=np.random.default_rng(1), scan_kwargs={"k": 75}, geometry=geometry
    )
    assert rep.p_value <= 0.05


def test_effect_sign_matches_lindo(ds, ctf):
    """Local 2SLS at the discovered cutoff recovers the sign of Lindo's +0.233.

    Documented deviation from the plan's literal recipe (estimate directly on
    the k=75 discovery): at detection bandwidth the single-neighborhood
    estimator is noise-dominated — n_used = 61 of 75 members and HC1
    SE ~= 0.36 against a true effect of ~0.23, so the sign of tau-hat is close
    to a coin flip (the observed k=75 value, -0.05 +/- 0.36, is well within
    1 SE of Lindo's estimate; nothing is wrong, there is just no power).
    Standard RD practice separates detection from estimation bandwidth, so the
    fine-stage discovery pins the LOCATION and the estimator then widens the
    local sample: a single-center rescan at k_est=150 around the discovered
    center feeds local_2sls. The result is stable in k_est (tau = +0.27 /
    +0.30 / +0.10 at k_est = 125 / 150 / 200, all with strong first stages),
    and 92% of the twelve best distinct near-cutoff neighborhoods at k=150
    give tau > 0.
    """
    near = [d for d in ctf.result.discoveries if abs(float(ds.Z[d.center_index, DIST])) < 0.1]
    assert near, "no fine-stage discovery within 0.1 GPA points of the cutoff"
    best = near[0]  # discoveries are LLR-sorted: first is the best near-cutoff one

    # Estimation-bandwidth rescan at the discovered center (location frozen,
    # neighborhood widened). It must still find the axis-aligned cutoff split,
    # otherwise the estimate below would not be the probation contrast.
    res = lord3_scan(ds, k=150, centers=np.array([best.center_index]))
    (wide,) = res.discoveries
    assert int(np.argmax(np.abs(wide.normal))) == DIST

    est = local_2sls(ds, wide)  # default outcome GPA_year2, ~13% NaN
    assert est.n_used >= 40  # NaN outcomes dropped inside the estimator, not zero-filled
    assert est.n_used < wide.members.size  # the NaN-y path was genuinely exercised
    assert 0.0 < est.tau < 1.0, f"tau = {est.tau} outside the Lindo bracket (0, 1)"
    assert est.weak_instrument is False
