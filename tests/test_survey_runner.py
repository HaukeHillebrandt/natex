"""Survey runner skeleton (phase survey task 5): SurveyResult, rdd/did, isolation.

Contract under test: ``survey()`` requires an explicit rng; always returns ALL
SEVEN families in FAMILY_ORDER with statuses from the fixed 5-value vocabulary
(credible|null|skipped|needs_input|failed); rdd runs end-to-end via
``natex.discover`` on the synthetic-shape CSV; a raising family runner is
isolated (status "failed", verbatim error, survey completes); per-family rng
sub-streams come from ONE upfront ``rng.spawn(7)`` so declaring an extra
family never shifts another family's stream; ``survey.json`` round-trips
through ``SurveyResult.load`` with an identical families dict.

Stochastic assertions: the rdd verdict on the synthetic CSV is only pinned to
{"credible", "null"} (both acceptable at q=9), and spawn stability is bitwise
equality under a fixed seed — checked across seeds 0/7 during implementation;
no margin needed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from natex.data.synthetic import make_synthetic
from natex.survey import SurveyResult, survey
from natex.survey.registry import FAMILY_ORDER

_BUDGET = {"q": 9, "k": 25}  # small explicit test budget (plan task 5)

_STATUSES = {"credible", "null", "skipped", "needs_input", "failed"}


def _write_synthetic_csv(root):
    """make_synthetic(n=300) binary-treatment CSV with a decoy binary column
    'holiday' inserted BEFORE T (recipe from tests/test_cli_study.py)."""
    ds, _ = make_synthetic(
        n=300, px=3, pz=2, zeta=6.0, kind="binary", rng=np.random.default_rng(0)
    )
    df = ds.df.copy()
    df.insert(df.columns.get_loc("T"), "holiday",
              np.random.default_rng(1).integers(0, 2, len(df)))
    path = root / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def _plain_cross_section(seed=0, n=200):
    """Pure rng normals x0..x3: no binary column, no panel, nothing to run."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])


def test_survey_requires_rng(tmp_path):
    df = _plain_cross_section()
    with pytest.raises(ValueError):
        survey(df, out_dir=tmp_path / "out")


def test_rdd_shape_end_to_end(tmp_path):
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    # ALL SEVEN families, fixed order.
    assert list(res.families) == list(FAMILY_ORDER)

    rdd = res.families["rdd"]
    assert rdd.status in {"credible", "null"}, rdd.reason
    assert np.isfinite(rdd.key_numbers["p_value"])
    # key_numbers is FLAT name->number: no nested dicts/lists (report contract)
    assert all(not isinstance(v, (dict, list)) for v in rdd.key_numbers.values())
    assert rdd.details_path == "families/rdd.json"
    assert (out / "families" / "rdd.json").exists()
    assert res.coverage["rdd"] is not None  # discover's searched block surfaced

    did = res.families["did"]
    assert did.status in {"skipped", "needs_input"}
    assert did.reason

    kink = res.families["kink"]
    assert kink.status == "needs_input"
    assert kink.reason == "no pre-declared cutoff (kink is candidate evaluation, not discovery)"

    path = out / "survey.json"
    assert path.exists()
    loaded = SurveyResult.load(path)
    assert loaded.families == res.families


def test_plain_cross_section_lists_all_seven(tmp_path):
    out = tmp_path / "out"
    res = survey(_plain_cross_section(), rng=np.random.default_rng(1), out_dir=out)
    assert list(res.families) == list(FAMILY_ORDER)
    for fam in res.families.values():
        assert fam.status in {"skipped", "needs_input"}  # none "failed"
        assert fam.reason
    assert len(res.coverage["not_run"]) == 7
    assert res.coverage["ran"] == []
    assert (out / "survey.json").exists()


def test_failure_isolation(monkeypatch, tmp_path):
    # NOTE: ``import natex.survey.runner`` would fail here — natex/__init__
    # binds the name ``survey`` to the function (discover precedent), so the
    # attribute path is shadowed; go through the subpackage module instead.
    from natex.survey import runner as runner_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom-xyzzy")

    monkeypatch.setattr(runner_mod, "_run_rdd", boom)
    csv = _write_synthetic_csv(tmp_path)
    out = tmp_path / "out"
    res = survey(str(csv), rng=np.random.default_rng(0), out_dir=out, budget=_BUDGET)

    rdd = res.families["rdd"]
    assert rdd.status == "failed"
    assert rdd.error == "boom-xyzzy"  # verbatim str(exc)
    assert "boom-xyzzy" in rdd.diagnostics["traceback"]
    # survey completed: all seven present, other families untouched
    assert list(res.families) == list(FAMILY_ORDER)
    assert res.families["did"].status in {"skipped", "needs_input"}
    assert res.families["kink"].status == "needs_input"
    assert (out / "survey.json").exists()


def test_spawn_stability(tmp_path):
    """Declaring a bunching threshold (an extra family) never shifts rdd's stream."""
    csv = _write_synthetic_csv(tmp_path)
    res_a = survey(str(csv), rng=np.random.default_rng(7), out_dir=tmp_path / "a",
                   budget=_BUDGET)
    res_b = survey(str(csv), rng=np.random.default_rng(7), out_dir=tmp_path / "b",
                   budget=_BUDGET, thresholds={"x0": 0.5})
    assert res_a.families["rdd"].key_numbers == res_b.families["rdd"].key_numbers
    assert res_a.families["rdd"].status == res_b.families["rdd"].status


def test_status_vocabulary(tmp_path):
    res = survey(_plain_cross_section(seed=3), rng=np.random.default_rng(4),
                 out_dir=tmp_path / "out")
    assert {f.status for f in res.families.values()} <= _STATUSES
    # survey.json statuses match too
    saved = json.loads((tmp_path / "out" / "survey.json").read_text())
    assert {f["status"] for f in saved["families"].values()} <= _STATUSES
