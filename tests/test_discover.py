"""natex.discover: ranked-first, exhaustive-still, coverage always reported (spec 6b)."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic
from natex.discover import discover, enumerate_configs
from natex.intake.plans import DesignCandidate, SearchPlan
from natex.llm import GuidanceResponse, MockBackend

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
    # no guidance backend => no log will ever be written: the path must be
    # None, not a phantom file reference (dogfood regression, Fitbit run)
    assert payload["guidance_log_path"] is None


def test_no_phantom_guidance_log_path_without_backend(tmp_path):
    """Dogfood regression (Fitbit run): a guidance-free ``out=`` run recorded
    ``out/guidance_log.jsonl`` in the report although the file never exists."""
    out = tmp_path / "out"
    rep = discover(_rdd_dataset(), rng=np.random.default_rng(1), budget=SMALL, out=out)
    assert rep.guidance_log_path is None
    assert not (out / "guidance_log.jsonl").exists()


def _did_binary_dataset():
    """Small panel with a BINARY treatment (the dogfood case that failed)."""
    rng = np.random.default_rng(3)
    n = 300
    codes = rng.integers(0, 3, size=(n, 2))
    t = rng.integers(0, 8, size=n).astype(float)
    p = 0.2 + 0.6 * ((codes[:, 0] == 1) & (t >= 4.0))
    df = pd.DataFrame(
        {
            "d0": codes[:, 0].astype(float),
            "d1": codes[:, 1].astype(float),
            "time": t,
            "T": (rng.random(n) < p).astype(float),
            "y": rng.normal(0.0, 1.0, n),
        }
    )
    spec = DatasetSpec(treatment="T", outcome="y", forcing=[],
                       covariates=["d0", "d1"], time="time")
    return Dataset(df, spec)


def test_did_binary_treatment_scans_under_default_budget(tmp_path):
    """Dogfood regression (Fitbit run): the budget defaults
    ``method='single_delta'`` + ``model='auto'`` failed every binary-treatment
    did config out of the box; the runner resolves auto -> normal (the remedy
    the error message itself prescribes)."""
    rep = discover(_did_binary_dataset(), design="did",
                   rng=np.random.default_rng(0),
                   budget={"q": 9, "bins": 3, "restarts": 2, "windows": (4.0,)})
    rec = rep.configs[0]
    assert rec.status == "scanned", rec.error
    assert rec.summary["design"] == "did"


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


# ---------------------------------------------------------------------------
# spec 6c: advisory in-scan guidance hooks — statistics NEVER gated or altered
# ---------------------------------------------------------------------------

INTERP = {"summary": "canned interpretation", "confounded_risk": "low"}
AUDIT_VETO = {"veto": True, "caveats": ["threshold looks confounded"]}
GESS_REVIEW = {"face_valid": False, "veto": True,
               "reason": "expanded into implausible profile"}
DID_BUDGET = {"q": 9, "bins": 3, "restarts": 2, "windows": (4.0,)}
_ADVISORY_KEYS = {"advisory", "advisory_veto", "vetoed_by_guidance"}


def _did_dataset():
    ds, _ = make_did_synthetic(n=300, d=2, V=3, zeta=8.0, rng=np.random.default_rng(1))
    return ds


def _strip_advisory(obj):
    """Remove every advisory-only key so guided vs unguided JSON can be compared."""
    if isinstance(obj, dict):
        return {k: _strip_advisory(v) for k, v in obj.items() if k not in _ADVISORY_KEYS}
    if isinstance(obj, list):
        return [_strip_advisory(v) for v in obj]
    return obj


class _VetoEverything:
    """Vetoes every hook — used to prove hooks never gate statistics."""

    name = "veto-everything"

    def complete(self, request):
        return GuidanceResponse(
            content={"veto": True, "face_valid": False, "reason": "always veto"},
            backend=self.name,
        )


class _SilentBackend:
    """Raises on every hook — used to prove hook failures never kill a config."""

    name = "silent"

    def complete(self, request):
        raise TimeoutError("agent silent")


@pytest.fixture(scope="module")
def rdd_hook_run(tmp_path_factory):
    mock = MockBackend([INTERP, AUDIT_VETO])
    out = tmp_path_factory.mktemp("rdd_hooks")
    rep = discover(_rdd_dataset(), guidance=mock, rng=np.random.default_rng(1),
                   budget=SMALL, out=out)
    return rep, mock, out


@pytest.fixture(scope="module")
def did_hook_run():
    mock = MockBackend([INTERP, {"veto": False}, GESS_REVIEW])
    rep = discover(_did_dataset(), design="did", guidance=mock,
                   rng=np.random.default_rng(0), budget=DID_BUDGET)
    base = discover(_did_dataset(), design="did",
                    rng=np.random.default_rng(0), budget=DID_BUDGET)
    return rep, base, mock


def test_mock_hooks_recorded_veto_is_flag_only(rdd_hook_run):
    rep, mock, _ = rdd_hook_run
    rec = rep.configs[0]
    assert rec.status == "scanned"
    assert rec.advisory["interpret_discovery"] == INTERP
    assert rec.advisory["audit_assumptions"] == AUDIT_VETO
    assert rec.advisory["vetoed"] is True
    assert rec.summary["advisory_veto"] is True
    # veto NEVER gates: effects still computed with finite tau
    assert np.isfinite(rec.summary["effects"]["2sls"]["tau"])
    # hook ORDER is a docstring contract for MockBackend users
    assert [r.task for r in mock.requests] == ["interpret_discovery", "audit_assumptions"]
    interp = mock.requests[0].payload
    assert set(interp) == {"candidate", "summary", "context"}
    assert "effects" not in interp["summary"]  # summary WITHOUT the effects key
    assert interp["context"] is None
    assert interp["candidate"]["design"] == "rdd"
    audit = mock.requests[1].payload
    assert set(audit) == {"candidate", "validation"}
    assert {"p_value", "placebo_passed", "placebo_holm", "density_p"} <= set(audit["validation"])


def test_never_gates_veto_everywhere_vs_none_identical_statistics():
    def run(guidance) -> dict:
        ds, _ = make_synthetic(n=300, zeta=6.0, kind="binary", rng=np.random.default_rng(0))
        return json.loads(
            discover(ds, guidance=guidance, rng=np.random.default_rng(7), budget=SMALL).to_json()
        )

    plain, vetoed = run(None), run(_VetoEverything())
    assert vetoed["configs"][0]["advisory"]["vetoed"] is True  # veto really fired
    assert _strip_advisory(plain) == _strip_advisory(vetoed)  # hooks consume no rng


def test_did_gess_review_veto_flag_only(did_hook_run):
    rep, base, mock = did_hook_run
    rec = rep.configs[0]
    assert rec.status == "scanned"
    gess = rec.summary["effects"]["gess"]
    assert gess["vetoed_by_guidance"] is True
    assert gess["tau"] == base.configs[0].summary["effects"]["gess"]["tau"]  # unchanged
    assert rec.advisory["control_review"] == GESS_REVIEW
    assert [r.task for r in mock.requests] == [
        "interpret_discovery", "audit_assumptions", "review_control_group"]
    assert set(mock.requests[2].payload) == {
        "profile", "expansions", "mse_trace", "subset_values", "n_control", "n_tau"}
    # audit responded veto=False: no flags set
    assert "vetoed" not in rec.advisory
    assert "advisory_veto" not in rec.summary
    # full did-path mutation check: identical JSON once advisory keys are stripped
    assert _strip_advisory(json.loads(base.to_json())) == _strip_advisory(
        json.loads(rep.to_json()))


def test_hook_error_isolated_config_still_scanned():
    rep = discover(_rdd_dataset(), guidance=_SilentBackend(),
                   rng=np.random.default_rng(1), budget=SMALL)
    rec = rep.configs[0]
    assert rec.status == "scanned"
    assert "agent silent" in rec.advisory["interpret_discovery"]["error"]
    assert "agent silent" in rec.advisory["audit_assumptions"]["error"]
    assert rec.llr is not None and rec.p_value is not None  # statistics untouched
    assert np.isfinite(rec.summary["effects"]["2sls"]["tau"])
    assert "advisory_veto" not in rec.summary  # errored hook can never veto


def _assert_no_outcome_key(obj, outcome: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert not (k in ("y", outcome) and isinstance(v, list)), f"outcome list under {k!r}"
            _assert_no_outcome_key(v, outcome)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_outcome_key(v, outcome)


def test_hook_payloads_carry_no_raw_outcome_values(rdd_hook_run, did_hook_run):
    for mock, ds in ((rdd_hook_run[1], _rdd_dataset()), (did_hook_run[2], _did_dataset())):
        y_reprs = [repr(float(v)) for v in ds.df[ds.spec.outcome].iloc[:5]]
        assert mock.requests
        for req in mock.requests:
            text = json.dumps(req.payload)  # also proves the payload is JSON-clean
            for y_repr in y_reprs:
                assert y_repr not in text
            _assert_no_outcome_key(req.payload, ds.spec.outcome)


def test_guidance_log_one_line_per_hook(rdd_hook_run):
    rep, mock, out = rdd_hook_run
    assert rep.guidance_log_path == str(out / "guidance_log.jsonl")
    lines = [json.loads(line)
             for line in (out / "guidance_log.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == len(mock.requests) == 2
    assert [e["task"] for e in lines] == ["interpret_discovery", "audit_assumptions"]
    assert all(e["backend"] == "mock" for e in lines)
