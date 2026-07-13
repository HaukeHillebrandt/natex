"""``ResultsBundle``: one results directory — results.json + figures/ + paper/.

The bundle is the contract between discovery and the paper pipeline (spec
sections 4 and 7): everything downstream (figures, templates, the research
brief) reads ONLY ``results.json``. Serialization goes through
:func:`natex.jsonutil.jsonable` exactly once (in :meth:`ResultsBundle.save`),
so NaN/inf become JSON null — never 0.0 (house rule). Coverage
(``searched``) is carried verbatim from the DiscoverReport (spec 6b).

No new inference code lives here except :func:`ivw_pooled`, a PRESENTATIONAL
fixed-effect combiner whose output must always be labeled indicative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from natex.data.spec import Dataset
from natex.discover import DiscoverReport
from natex.intake.analyst import IntakeReport
from natex.jsonutil import jsonable

BUNDLE_SCHEMA = 1  # top-level "natex_bundle" marker distinguishes bundle vs raw scan payloads


@dataclass(frozen=True)
class PooledEffect:
    tau: float  # NaN when no usable input — never 0.0
    se: float  # NaN when no usable input
    ci: tuple[float, float]
    n_used: int


def ivw_pooled(tau, se) -> PooledEffect:
    """Fixed-effect inverse-variance pooling; PRESENTATIONAL ONLY.

    Drops entries where tau or se is non-finite or se <= 0. weights = 1/se^2;
    pooled se = sqrt(1/sum w); ci = tau +/- 1.96 se. All-NaN input -> PooledEffect
    with NaN tau/se/ci and n_used=0. Callers must label the pooled row as
    indicative (the per-estimator inputs are NOT independent).
    """
    t = np.asarray(tau, dtype=float).ravel()
    s = np.asarray(se, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError(f"tau and se must have equal length, got {t.size} and {s.size}")
    ok = np.isfinite(t) & np.isfinite(s) & (s > 0)
    n_used = int(ok.sum())
    if n_used == 0:
        nan = float("nan")
        return PooledEffect(tau=nan, se=nan, ci=(nan, nan), n_used=0)
    w = 1.0 / s[ok] ** 2
    pooled = float(np.sum(w * t[ok]) / np.sum(w))
    pooled_se = float(np.sqrt(1.0 / np.sum(w)))
    ci = (pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se)
    return PooledEffect(tau=pooled, se=pooled_se, ci=ci, n_used=n_used)


def _data_block(dataset: Dataset | None, report: DiscoverReport) -> dict | None:
    """The ``data`` block: from the bound spec when given, else the best (or
    first) config's candidate with ``n_rows`` null — never fabricated."""
    if dataset is not None:
        spec = dataset.spec
        return {
            "n_rows": len(dataset.df),
            # Issue #1: rows used alone hides listwise deletion — surface the
            # input size and the top per-column attributable losses.
            "n_rows_input": dataset.n_rows_input,
            "row_loss": dataset.top_row_loss(),
            "treatment": spec.treatment,
            "outcome": spec.outcome,
            "forcing": list(spec.forcing),
            "time": spec.time,
            "unit": spec.unit,
            "covariates": list(spec.covariates),
            "source": None,
        }
    rec = report.best()
    if rec is None and report.configs:
        rec = report.configs[0]
    if rec is None:
        return None
    c = rec.candidate
    return {
        "n_rows": None,
        "n_rows_input": None,
        "row_loss": None,
        "treatment": c.treatment,
        "outcome": c.outcome,
        "forcing": list(c.forcing),
        "time": c.time,
        "unit": c.unit,
        "covariates": None,
        "source": None,
    }


def _intake_block(intake: IntakeReport | None) -> dict | None:
    if intake is None:
        return None
    return {
        "source": intake.source,
        "context": intake.context,
        "understanding": intake.understanding.model_dump(),
        "guidance_errors": list(intake.guidance_errors),
        "prep_log": list(intake.prep_log),
    }


def _scan_payload_config(payload: dict) -> dict | None:
    """One ConfigRecord-shaped view of a plain single-scan results.json.

    The non-plan ``natex discover`` payload (rdd: params/scan/discoveries/
    validation/effects; did: the same nested under ``"did"``) becomes a single
    ``status="scanned"`` config so ``_paper_context``/``_scanned_configs`` —
    and therefore ``natex paper`` / ``natex brief`` — render the run (F-D1).
    Returns None when the payload has neither shape (nothing is fabricated).
    """
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if isinstance(payload.get("did"), dict):
        did = payload["did"]
        scan = did.get("scan") or {}
        discoveries = did.get("discoveries") or []
        validation = did.get("validation") or {}
        top = discoveries[0] if discoveries else {}
        candidate = {
            "design": "did",
            "treatment": params.get("treatment"),
            "outcome": params.get("outcome"),
            "forcing": params.get("forcing") or [],
            "unit": params.get("unit"),
            "time": params.get("time"),
        }
        summary = {
            "design": "did",
            "subset_values": top.get("subset_values"),
            "t0": top.get("t0"),
            "window": top.get("window"),
            "null_kind": scan.get("null_kind"),
            "composition_passed": validation.get("composition_passed"),
            "anticipation_passed": validation.get("anticipation_passed"),
            "effects": did.get("effects") or {},
        }
    elif isinstance(payload.get("scan"), dict):
        scan = payload["scan"]
        discoveries = payload.get("discoveries") or []
        validation = payload.get("validation") or {}
        top = discoveries[0] if discoveries else {}
        influence = top.get("forcing_influence") or {}
        candidate = {
            "design": "rdd",
            "treatment": params.get("treatment"),
            "outcome": params.get("outcome"),
            # Issue #29: prefer the params-recorded forcing; the top
            # discovery's forcing_influence keys remain the fallback for
            # pre-fix payloads (and vanish when there are no discoveries).
            "forcing": params.get("forcing") or list(influence),
        }
        summary = {
            "design": "rdd",
            "center_z": top.get("center_z"),
            "normal": top.get("normal"),
            "forcing_influence": influence,
            "placebo_passed": validation.get("placebo_passed"),
            "placebo_holm": validation.get("placebo_holm"),
            "density_p": validation.get("density_p"),
            "effects": payload.get("effects") or {},
        }
        if payload.get("coarse") is not None:
            summary["coarse"] = payload["coarse"]
    else:
        return None
    return {
        "candidate": candidate,
        "source": "scan",
        "status": "scanned",
        "llr": scan.get("observed_max_llr"),
        "p_value": scan.get("p_value"),
        "n_discoveries": len(discoveries),
        "summary": summary,
    }


class ResultsBundle:
    """A results directory: results.json + figures/ + paper/ (spec section 7)."""

    def __init__(self, out_dir: str | Path, results: dict):
        self.out_dir = Path(out_dir)
        self.results = results  # JSON-native after save()/load(); see save()

    @property
    def results_path(self) -> Path:
        return self.out_dir / "results.json"

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def paper_dir(self) -> Path:
        return self.out_dir / "paper"

    @classmethod
    def from_discover(
        cls,
        report: DiscoverReport,
        out_dir: str | Path,
        *,
        dataset: Dataset | None = None,
        intake: IntakeReport | None = None,
        seed: int | None = None,
        params: dict | None = None,
    ) -> ResultsBundle:
        import natex  # lazy: natex/__init__ imports this module

        guidance_log_path = report.guidance_log_path
        if guidance_log_path is None and intake is not None:
            guidance_log_path = intake.guidance_log_path
        results = {
            "natex_bundle": BUNDLE_SCHEMA,
            "natex_version": natex.__version__,
            "created": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "params": params if params is not None else report.searched.get("budget"),
            "searched": report.searched,  # coverage verbatim (spec 6b)
            "configs": [rec.to_dict() for rec in report.configs],
            "best_index": report.best_index,
            "guidance_log_path": guidance_log_path,
            "data": _data_block(dataset, report),
            "intake": _intake_block(intake),
            "figures": [],
        }
        return cls(out_dir, results)

    @classmethod
    def from_scan_payload(cls, payload: dict, out_dir: str | Path) -> ResultsBundle:
        """Wrap a single-scan results.json payload (the non-plan ``natex
        discover`` schema: params/scan/discoveries/validation/effects, or the
        did variant under ``"did"``) under ``{"scan": payload}``, lifting seed
        from ``payload["params"]["seed"]`` when present, AND adapt it into a
        one-config view (``configs``/``best_index``) so the paper and brief
        renderers show the run instead of an empty document (finding F-D1).
        Provenance fields it cannot know stay null."""
        params = payload.get("params")
        seed = params.get("seed") if isinstance(params, dict) else None
        config = _scan_payload_config(payload)
        results = {
            "natex_bundle": BUNDLE_SCHEMA,
            "natex_version": None,
            "created": None,
            "seed": seed,
            "params": params,
            "searched": None,
            "configs": [config] if config is not None else None,
            "best_index": 0 if config is not None else None,
            "guidance_log_path": None,
            "data": None,
            "intake": None,
            "figures": [],
            "scan": payload,
        }
        return cls(out_dir, results)

    @classmethod
    def load(cls, dir: str | Path) -> ResultsBundle:
        """Load a bundle directory. Resolution order:

        1. ``dir/results.json`` with the ``"natex_bundle"`` key -> bundle as saved;
        2. ``dir/results.json`` WITHOUT the marker -> :meth:`from_scan_payload`;
        3. ``dir/discover_report.json`` -> adapt the DiscoverReport JSON
           (configs/searched/best_index/guidance_log_path verbatim;
           version/seed/params null);
        4. else FileNotFoundError naming every path it looked for.
        """
        d = Path(dir)
        results_path = d / "results.json"
        report_path = d / "discover_report.json"
        if results_path.exists():
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            if "natex_bundle" in payload:
                return cls(d, payload)
            return cls.from_scan_payload(payload, d)
        if report_path.exists():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            results = {
                "natex_bundle": BUNDLE_SCHEMA,
                "natex_version": None,
                "created": None,
                "seed": None,
                "params": None,
                "searched": rep.get("searched"),
                "configs": rep.get("configs", []),
                "best_index": rep.get("best_index"),
                "guidance_log_path": rep.get("guidance_log_path"),
                "data": None,
                "intake": None,
                "figures": [],
            }
            return cls(d, results)
        raise FileNotFoundError(
            f"no results bundle in {d}: looked for {results_path} (bundle or "
            f"single-scan payload) and {report_path}"
        )

    def add_figure(self, name: str, png: Path, pdf: Path) -> None:
        """Append/replace (by name) a manifest entry in ``results["figures"]``;
        paths stored POSIX-relative to ``out_dir``."""

        def _rel(p: Path) -> str:
            p = Path(p)
            try:
                return p.relative_to(self.out_dir).as_posix()
            except ValueError:
                return p.as_posix()  # outside the bundle: keep the path as given

        entry = {"name": name, "png": _rel(png), "pdf": _rel(pdf)}
        figures = self.results.setdefault("figures", [])
        for i, f in enumerate(figures):
            if f.get("name") == name:
                figures[i] = entry
                return
        figures.append(entry)

    def save(self) -> Path:
        """mkdir ``out_dir``, ``figures/``, ``paper/``; write results.json.

        ``jsonable`` runs here exactly once (NaN/inf -> null) and the coerced
        dict replaces ``self.results``, so the in-memory bundle round-trips
        bitwise against :meth:`load`.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        self.paper_dir.mkdir(exist_ok=True)
        self.results = jsonable(self.results)
        self.results_path.write_text(
            json.dumps(self.results, indent=1), encoding="utf-8"
        )
        return self.results_path
