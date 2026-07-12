"""natex.discover: ranked-first, exhaustive-still, coverage always reported (spec 6b)."""

import importlib
import json

import numpy as np
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic
from natex.discover import discover, enumerate_configs
from natex.intake.plans import DesignCandidate, SearchPlan

# `from natex.discover import ...` resolves the submodule, but attribute access
# on the package (`natex.discover`) returns the FUNCTION after __init__ rebinds
# the name — monkeypatching must target the real module object.
discover_mod = importlib.import_module("natex.discover")

SMALL = {"k": 25, "q": 9}  # every scan in this file: n <= 300, q = 9, seeded


def _rdd_dataset():
    ds, _ = make_synthetic(n=300, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
    return ds


def _rdd_dataset_with_decoy():
    """The rdd synthetic plus a decoy binary column usable as a second treatment."""
    ds = _rdd_dataset()
    df = ds.df.copy()
    df["holiday"] = (np.arange(len(df)) % 2).astype(float)
    spec = DatasetSpec(
        treatment="T", outcome="y", forcing=["x0", "x1"],
        covariates=["x0", "x1", "holiday"],
    )
    return Dataset(df, spec)


def _two_candidate_plan():
    return SearchPlan(
        candidates=[
            DesignCandidate(design="rdd", treatment="T", outcome="y",
                            forcing=["x0", "x1"], priority=0),
            DesignCandidate(design="rdd", treatment="holiday", outcome="y",
                            forcing=["x0", "x1"], priority=1),
        ]
    )


# ---------------------------------------------------------------------------
# argument contract
# ---------------------------------------------------------------------------


def test_rng_required():
    with pytest.raises(ValueError, match="Generator"):
        discover(_rdd_dataset())


def test_bad_design_named_in_error():
    with pytest.raises(ValueError, match="banana"):
        discover(_rdd_dataset(), design="banana", rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="banana"):
        enumerate_configs(_rdd_dataset(), design="banana")


def test_unknown_budget_key_named_in_error():
    with pytest.raises(ValueError, match="warp_speed"):
        discover(_rdd_dataset(), rng=np.random.default_rng(0), budget={"warp_speed": 1})


def test_unknown_search_plan_budget_key_named_in_error():
    plan = SearchPlan(candidates=[], budget={"warp_speed": 1})
    with pytest.raises(ValueError, match="warp_speed"):
        discover(_rdd_dataset(), search_plan=plan, rng=np.random.default_rng(0))


def test_budget_precedence_explicit_wins_over_plan_hints():
    plan = SearchPlan(
        candidates=[DesignCandidate(design="rdd", treatment="T", outcome="y",
                                    forcing=["x0", "x1"])],
        budget={"k": 40, "q": 9},
    )
    rep = discover(_rdd_dataset(), search_plan=plan,
                   rng=np.random.default_rng(1), budget={"k": 25})
    assert rep.searched["budget"]["k"] == 25  # explicit arg wins
    assert rep.searched["budget"]["q"] == 9  # plan hint kept
    assert rep.searched["budget"]["degree"] == 1  # default kept


# ---------------------------------------------------------------------------
# rdd auto path
# ---------------------------------------------------------------------------


def test_rdd_auto_single_config(tmp_path):
    rep = discover(_rdd_dataset(), rng=np.random.default_rng(1), budget=SMALL,
                   out=tmp_path / "out")
    assert len(rep.configs) == 1
    rec = rep.configs[0]
    assert rec.source == "exhaustive"
    assert rec.status == "scanned"
    assert rec.llr is not None and rec.llr > 0.0
    assert rec.p_value is not None and 0.0 < rec.p_value <= 1.0
    assert rec.n_discoveries >= 1
    assert rec.summary["design"] == "rdd"
    assert set(rec.summary["forcing_influence"]) == {"x0", "x1"}
    assert rep.best() is rec

    payload = json.loads(rep.to_json())
    searched = payload["searched"]
    assert searched["n_total"] == 1
    assert searched["n_scanned"] == 1
    assert searched["n_skipped_budget"] == 0
    assert searched["n_failed"] == 0
    assert searched["n_invalid"] == 0
    assert searched["budget"]["k"] == 25
    assert searched["plan_candidates"] == 0
    assert searched["exhaustive_candidates"] == 1
    assert (tmp_path / "out" / "discover_report.json").exists()
    assert payload["guidance_log_path"].endswith("guidance_log.jsonl")


# ---------------------------------------------------------------------------
# spec 6b: plan orders the scan, never truncates it
# ---------------------------------------------------------------------------


def test_plan_candidates_scanned_first_and_exhaustive_deduped():
    rep = discover(_rdd_dataset_with_decoy(), search_plan=_two_candidate_plan(),
                   rng=np.random.default_rng(2), budget=SMALL)
    # priority-0 plan candidate first; the exhaustive (T, x0, x1) config is
    # absorbed by the identical plan candidate on DesignCandidate.key().
    assert [r.candidate.treatment for r in rep.configs] == ["T", "holiday"]
    assert rep.configs[0].source == "plan"
    assert rep.configs[1].source == "plan"
    assert rep.searched["n_total"] == 2
    assert rep.searched["plan_candidates"] == 2
    assert rep.searched["exhaustive_candidates"] == 0
    assert all(r.status == "scanned" for r in rep.configs)
    assert rep.best().candidate.treatment == "T"  # planted jump beats the decoy


def test_max_configs_skips_are_still_listed():
    rep = discover(_rdd_dataset_with_decoy(), search_plan=_two_candidate_plan(),
                   rng=np.random.default_rng(2), budget={**SMALL, "max_configs": 1})
    assert rep.configs[0].status == "scanned"
    assert rep.configs[1].status == "skipped_budget"  # cut, NOT dropped (spec 6b)
    assert rep.searched["n_skipped_budget"] == 1
    assert rep.searched["n_total"] == 2
    payload = json.loads(rep.to_json())
    skipped = payload["configs"][1]
    assert skipped["status"] == "skipped_budget"
    assert skipped["llr"] is None
    assert skipped["p_value"] is None


# ---------------------------------------------------------------------------
# invalid / failed isolation
# ---------------------------------------------------------------------------


def test_invalid_plan_candidate_recorded_not_dropped():
    plan = SearchPlan(candidates=[
        DesignCandidate(design="rdd", treatment="nonexistent_col", outcome="y",
                        forcing=["x0", "x1"], priority=0),
    ])
    rep = discover(_rdd_dataset(), search_plan=plan,
                   rng=np.random.default_rng(3), budget=SMALL)
    inv = rep.configs[0]
    assert inv.status == "invalid"
    assert "nonexistent_col" in inv.error
    assert inv.llr is None and inv.p_value is None
    scanned = [r for r in rep.configs if r.status == "scanned"]
    assert len(scanned) == 1 and scanned[0].candidate.treatment == "T"
    assert rep.searched["n_invalid"] == 1
    assert rep.searched["n_total"] == 2


def test_failed_config_isolated_and_llr_p_null_in_json(monkeypatch):
    real = discover_mod.lord3_scan
    calls = {"n": 0}

    def flaky(ds, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic scan failure")
        return real(ds, **kwargs)

    monkeypatch.setattr(discover_mod, "lord3_scan", flaky)
    rep = discover(_rdd_dataset_with_decoy(), search_plan=_two_candidate_plan(),
                   rng=np.random.default_rng(4), budget=SMALL)
    first, second = rep.configs
    assert first.status == "failed"
    assert "synthetic scan failure" in first.error
    assert second.status == "scanned"  # a failed config never kills the sweep
    assert rep.searched["n_failed"] == 1
    payload = json.loads(rep.to_json())
    assert payload["configs"][0]["llr"] is None  # NaN policy: None, never 0.0
    assert payload["configs"][0]["p_value"] is None
    assert rep.best() is second


# ---------------------------------------------------------------------------
# did path
# ---------------------------------------------------------------------------


def test_did_path_summary_and_effects():
    ds, _ = make_did_synthetic(n=300, d=2, V=3, zeta=8.0, rng=np.random.default_rng(1))
    rep = discover(ds, design="did", rng=np.random.default_rng(0),
                   budget={"q": 9, "bins": 3, "restarts": 2, "windows": (4.0,)})
    assert rep.searched["n_total"] == 1
    rec = rep.configs[0]
    assert rec.status == "scanned"
    assert rec.llr is not None and rec.llr > 0.0
    assert rec.p_value is not None and 0.0 < rec.p_value <= 1.0
    s = rec.summary
    assert s["design"] == "did"
    assert isinstance(s["subset_values"], dict) and s["subset_values"]
    assert set(s["effects"]) == {"dd", "synthetic", "gess"}
    for block in s["effects"].values():
        assert {"tau", "se", "p", "pre_mse", "dose"} <= set(block)
    assert s["searched_windows"] == [4.0]
    assert s["restarts"] == 2
    assert s["null_kind"] in ("ar1_unit", "iid", "bernoulli")
    json.loads(rep.to_json())  # serializable (NaN -> null, never 0)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_identical_seed_bitwise_identical_json():
    def run() -> str:
        ds, _ = make_synthetic(n=300, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
        return discover(ds, rng=np.random.default_rng(7), budget=SMALL).to_json()

    assert run() == run()
