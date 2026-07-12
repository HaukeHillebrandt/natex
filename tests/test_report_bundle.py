"""ResultsBundle: results.json with version/seed/coverage metadata (spec 7, 6b)."""

import json

import numpy as np
import pytest

import natex
from natex.report.bundle import ResultsBundle, ivw_pooled
from report_helpers import make_did_bundle, make_rdd_bundle

# ---------------------------------------------------------------------------
# session-scoped seeded runs shared across assertions (discover is the slow bit)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rdd(tmp_path_factory):
    return make_rdd_bundle(tmp_path_factory.mktemp("rdd_bundle"))


@pytest.fixture(scope="module")
def did(tmp_path_factory):
    return make_did_bundle(tmp_path_factory.mktemp("did_bundle"))


# ---------------------------------------------------------------------------
# 1. save() writes results.json + figures/ + paper/
# ---------------------------------------------------------------------------


def test_save_writes_layout(rdd):
    bundle, _, _ = rdd
    assert bundle.results_path.exists()
    assert bundle.figures_dir.is_dir()
    assert bundle.paper_dir.is_dir()
    assert bundle.results_path == bundle.out_dir / "results.json"


def test_did_bundle_layout(did):
    bundle, _, _ = did
    assert bundle.results_path.exists()
    assert bundle.figures_dir.is_dir()
    assert bundle.paper_dir.is_dir()


# ---------------------------------------------------------------------------
# 2. metadata: version, seed, coverage verbatim, finite best effects
# ---------------------------------------------------------------------------


def test_metadata_and_coverage(rdd):
    bundle, report, _ = rdd
    r = bundle.results
    assert r["natex_bundle"] == 1
    assert r["natex_version"] == natex.__version__
    assert r["seed"] == 0
    assert r["searched"] == json.loads(report.to_json())["searched"]
    best = r["configs"][r["best_index"]]
    tau = best["summary"]["effects"]["2sls"]["tau"]
    assert isinstance(tau, float) and np.isfinite(tau)


def test_data_block_from_dataset(rdd):
    bundle, _, ds = rdd
    d = bundle.results["data"]
    assert d["n_rows"] == len(ds.df)
    assert d["treatment"] == "T"
    assert d["outcome"] == "y"
    assert d["forcing"] == ["x0", "x1"]
    assert d["time"] is None


def test_did_metadata(did):
    bundle, report, ds = did
    r = bundle.results
    assert r["searched"] == json.loads(report.to_json())["searched"]
    assert r["data"]["time"] == "t"
    assert r["data"]["treatment"] == "theta"
    best = r["configs"][r["best_index"]]
    assert set(best["summary"]["effects"]) == {"dd", "synthetic", "gess"}


# ---------------------------------------------------------------------------
# 3. round-trip: load(dir).results == saved.results (JSON-native)
# ---------------------------------------------------------------------------


def test_round_trip_exact(rdd):
    bundle, _, _ = rdd
    loaded = ResultsBundle.load(bundle.out_dir)
    assert loaded.results == bundle.results
    assert loaded.out_dir == bundle.out_dir


# ---------------------------------------------------------------------------
# 4. NaN -> null, never "NaN" in the file, never 0.0
# ---------------------------------------------------------------------------


def test_nan_becomes_null(tmp_path):
    bundle, _, _ = make_rdd_bundle(tmp_path)
    best = bundle.results["configs"][bundle.results["best_index"]]
    best["summary"]["density_p"] = float("nan")
    bundle.save()
    text = bundle.results_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    reloaded = ResultsBundle.load(tmp_path)
    rebest = reloaded.results["configs"][reloaded.results["best_index"]]
    assert rebest["summary"]["density_p"] is None


# ---------------------------------------------------------------------------
# 5. load() adaptation: single-scan payload, discover_report.json, empty dir
# ---------------------------------------------------------------------------


def test_load_adapts_single_scan_payload(tmp_path):
    payload = {
        "params": {"k": 25, "q": 9, "seed": 7},
        "scan": {"model": "normal", "p_value": 0.1, "observed_max_llr": 3.0},
        "discoveries": [],
        "validation": {"placebo_passed": True},
        "effects": {},
    }
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    bundle = ResultsBundle.load(tmp_path)
    assert bundle.results["natex_bundle"] == 1
    assert bundle.results["scan"] == payload
    assert bundle.results["seed"] == 7
    assert bundle.results["natex_version"] is None


def test_load_adapts_discover_report(tmp_path, rdd):
    _, report, _ = rdd
    report.save(tmp_path)  # writes ONLY discover_report.json
    bundle = ResultsBundle.load(tmp_path)
    ref = json.loads(report.to_json())
    assert bundle.results["natex_version"] is None
    assert bundle.results["seed"] is None
    assert bundle.results["params"] is None
    assert bundle.results["configs"] == ref["configs"]
    assert bundle.results["searched"] == ref["searched"]
    assert bundle.results["best_index"] == ref["best_index"]


def test_load_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        ResultsBundle.load(tmp_path)
    msg = str(err.value)
    assert "results.json" in msg
    assert "discover_report.json" in msg


# ---------------------------------------------------------------------------
# add_figure manifest
# ---------------------------------------------------------------------------


def test_add_figure_manifest(tmp_path):
    bundle, _, _ = make_rdd_bundle(tmp_path)
    png = bundle.figures_dir / "scatter.png"
    pdf = bundle.figures_dir / "scatter.pdf"
    png.write_bytes(b"png")
    pdf.write_bytes(b"pdf")
    bundle.add_figure("scatter", png, pdf)
    assert bundle.results["figures"] == [
        {"name": "scatter", "png": "figures/scatter.png", "pdf": "figures/scatter.pdf"}
    ]
    # replace by name, never duplicate
    bundle.add_figure("scatter", png, pdf)
    assert len(bundle.results["figures"]) == 1
    bundle.save()
    assert ResultsBundle.load(tmp_path).results["figures"] == bundle.results["figures"]


# ---------------------------------------------------------------------------
# 6. ivw_pooled: presentational combiner, NaN never 0.0
# ---------------------------------------------------------------------------


def test_ivw_pooled_two_equal_weights():
    p = ivw_pooled([1.0, 3.0], [1.0, 1.0])
    assert p.tau == 2.0
    assert p.se == pytest.approx(1 / np.sqrt(2))
    assert p.ci == pytest.approx((2 - 1.96 / np.sqrt(2), 2 + 1.96 / np.sqrt(2)))
    assert p.n_used == 2


def test_ivw_pooled_drops_nonfinite():
    p = ivw_pooled([1.0, np.nan], [1.0, 1.0])
    assert p.n_used == 1
    assert p.tau == 1.0


def test_ivw_pooled_all_nan_stays_nan():
    p = ivw_pooled([np.nan], [np.nan])
    assert p.n_used == 0
    assert np.isnan(p.tau)  # never 0.0
    assert np.isnan(p.se)
    assert np.isnan(p.ci[0]) and np.isnan(p.ci[1])


def test_ivw_pooled_rejects_nonpositive_se():
    p = ivw_pooled([1.0, 5.0], [1.0, 0.0])
    assert p.n_used == 1
    assert p.tau == 1.0


# ---------------------------------------------------------------------------
# 7. intake= wiring (NullBackend study path, no LLM)
# ---------------------------------------------------------------------------


def test_intake_wiring(tmp_path):
    from report_helpers import SMALL

    ds, _ = natex.make_did_synthetic(n=400, d=2, V=3, zeta=8.0, rng=np.random.default_rng(0))
    csv = tmp_path / "panel.csv"
    ds.df.to_csv(csv, index=False)
    intake = natex.study(csv, rng=np.random.default_rng(0), out=tmp_path / "intake")
    report = natex.discover(ds, rng=np.random.default_rng(0), budget=SMALL)
    bundle = ResultsBundle.from_discover(
        report, tmp_path / "bundle", dataset=ds, intake=intake, seed=0
    )
    bundle.save()
    r = ResultsBundle.load(tmp_path / "bundle").results
    assert "shape" in r["intake"]["understanding"]
    assert r["guidance_log_path"] is not None


# ---------------------------------------------------------------------------
# 8. no-outcome bundle: effects {}, still saves and loads
# ---------------------------------------------------------------------------


def test_no_outcome_bundle(tmp_path):
    bundle, _, ds = make_rdd_bundle(tmp_path, with_outcome=False)
    assert ds.y is None
    best = bundle.results["configs"][bundle.results["best_index"]]
    assert best["summary"]["effects"] == {}
    assert bundle.results["data"]["outcome"] is None
    loaded = ResultsBundle.load(tmp_path)
    assert loaded.results == bundle.results


# ---------------------------------------------------------------------------
# top-level export
# ---------------------------------------------------------------------------


def test_top_level_export():
    assert natex.ResultsBundle is ResultsBundle
