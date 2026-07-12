"""paperbanana method-diagram adapter (phase report-paper task 7).

No network, ever: the installed-library path monkeypatches a FAKE
``paperbanana`` module into ``sys.modules`` (recorder that writes a 1-byte
file); the missing-library path patches the entry to None. The description
builder is pure text and runs on a core-only install.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from natex.report.paperbanana import _pipeline_description, generate_method_diagram
from report_helpers import make_did_bundle, make_rdd_bundle


@pytest.fixture(scope="module")
def rdd(tmp_path_factory):
    return make_rdd_bundle(tmp_path_factory.mktemp("rdd_banana"))


@pytest.fixture(scope="module")
def did(tmp_path_factory):
    return make_did_bundle(tmp_path_factory.mktemp("did_banana"))


# ---------------------------------------------------------------------------
# 1. _pipeline_description: pure, deterministic, names the pipeline as run
# ---------------------------------------------------------------------------


def test_description_rdd_names_pipeline(rdd):
    bundle, report, _ = rdd
    text = _pipeline_description(bundle)
    assert "LoRD3" in text
    assert str(int(report.searched["n_scanned"])) in text
    assert str(int(report.searched["n_total"])) in text
    assert "randomization" in text
    assert "2SLS" in text


def test_description_did_names_suddds(did):
    bundle, _, _ = did
    assert "SuDDDS" in _pipeline_description(bundle)


def test_description_deterministic(rdd):
    bundle, _, _ = rdd
    assert _pipeline_description(bundle) == _pipeline_description(bundle)


# ---------------------------------------------------------------------------
# 2. generate_method_diagram: documented single call contract vs fake module
# ---------------------------------------------------------------------------


def test_generate_method_diagram_call_contract(rdd, tmp_path, monkeypatch):
    bundle, _, _ = rdd
    out = tmp_path / "method_diagram.png"
    recorded = {}

    def recorder(*, description, output_path):
        recorded["description"] = description
        recorded["output_path"] = output_path
        Path(output_path).write_bytes(b"x")
        return output_path

    monkeypatch.setitem(
        sys.modules, "paperbanana", types.SimpleNamespace(generate_diagram=recorder)
    )
    result = generate_method_diagram(bundle, out)
    assert result == out
    assert out.exists()
    assert "LoRD3" in recorded["description"]
    assert recorded["output_path"] == str(out)


def test_generate_method_diagram_none_result_falls_back_to_out(
    rdd, tmp_path, monkeypatch
):
    bundle, _, _ = rdd
    out = tmp_path / "diagram.png"

    def writer(*, description, output_path):
        Path(output_path).write_bytes(b"x")
        return None  # library returns nothing -> adapter falls back to out

    monkeypatch.setitem(
        sys.modules, "paperbanana", types.SimpleNamespace(generate_diagram=writer)
    )
    assert generate_method_diagram(bundle, out) == out


# ---------------------------------------------------------------------------
# 3. import guard: module imports core-clean; call names the paperbanana extra
# ---------------------------------------------------------------------------


def test_generate_method_diagram_missing_extra(rdd, tmp_path, monkeypatch):
    bundle, _, _ = rdd
    monkeypatch.setitem(sys.modules, "paperbanana", None)
    with pytest.raises(ImportError, match=r"natex-discovery\[paperbanana\]"):
        generate_method_diagram(bundle, tmp_path / "d.png")


def test_module_import_without_paperbanana(monkeypatch):
    monkeypatch.setitem(sys.modules, "paperbanana", None)
    monkeypatch.delitem(sys.modules, "natex.report.paperbanana", raising=False)
    import natex.report.paperbanana  # noqa: F401  (must not touch paperbanana)
