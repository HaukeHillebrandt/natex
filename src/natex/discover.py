"""``natex.discover()``: budget-aware discovery orchestration (spec 6b).

A :class:`~natex.intake.plans.SearchPlan` ORDERS the scan, it never truncates
it: ranked plan candidates run first at full resolution, the exhaustive
remainder (derived from the bound dataset spec) still runs within budget, and
the report ALWAYS records what was and wasn't searched — every enumerated
configuration becomes a :class:`ConfigRecord` with status ``scanned``,
``skipped_budget``, ``failed`` or ``invalid``. Budget cuts and per-config
failures are recorded, never dropped, and never fabricate numbers
(``llr``/``p_value`` stay ``None``; NaN serializes to null, never 0).

Statistics are reused verbatim from the scan/validate/estimate modules —
this module adds NO new inference code (audit 1 lineage). Discovery never
reads the outcome; effects are estimated only when the candidate names one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec
from natex.did.controls import gess_control
from natex.did.effects import did_effect, tau_randomization_test
from natex.did.panel import build_panel
from natex.did.suddds import resolve_default_model, suddds_scan
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.intake.plans import DesignCandidate, SearchPlan
from natex.jsonutil import jsonable
from natex.llm import GuidanceBackend, GuidanceLog, GuidanceRequest, LoggedBackend
from natex.rdd.lord3 import lord3_scan
from natex.scan.coarse import coarse_to_fine_scan, coarse_to_fine_search
from natex.scan.geometry import build_geometry
from natex.validate.density import density_test
from natex.validate.panel import (
    anticipation_test,
    composition_test,
    panel_randomization_test,
)
from natex.validate.placebo import placebo_tests
from natex.validate.randomization import randomization_test

_BUDGET_DEFAULTS = {"max_configs": None, "k": 50, "q": 99, "degree": 1, "coarse": False,
                    "n_coarse": 2000, "bins": 4, "restarts": 8, "method": "single_delta",
                    "model": "auto", "windows": None}

_DESIGNS = ("auto", "rdd", "did")

# Per-config failures that are isolated (status="failed", sweep continues);
# never a bare except — programming errors still propagate.
_CONFIG_EXCEPTIONS = (ValueError, RuntimeError, np.linalg.LinAlgError)

# Guidance-hook failures that become advisory errors (spec 6c): the config
# stays scanned and its statistics are untouched. Never a bare except.
_HOOK_EXCEPTIONS = (TimeoutError, ValueError, RuntimeError)


class _GuidanceHooks:
    """Advisory in-scan hooks for ONE config (spec 6c), no-ops without a backend.

    Order per scanned config (contract for MockBackend response lists):
    ``interpret_discovery`` -> ``audit_assumptions`` -> (did GESS path only)
    ``review_control_group``. Guidance proposes and vetoes but NEVER gates or
    alters statistics — a veto is only ever a flag in the output. Payloads are
    coerced with :func:`jsonable` and carry only summary statistics already
    destined for the results bundle: no raw data arrays, no outcome values
    (audit "discovery never reads y" lineage). A hook failure in
    ``_HOOK_EXCEPTIONS`` is recorded as ``advisory[<hook>] = {"error": ...}``.
    """

    def __init__(self, guidance: GuidanceBackend | None,
                 candidate: DesignCandidate, advisory: dict):
        self._guidance = guidance
        self._candidate = candidate
        self._advisory = advisory

    def _call(self, task: str, payload: dict, key: str) -> dict | None:
        """Fire one hook; returns its content, or None (no backend / failure)."""
        if self._guidance is None:
            return None
        request = GuidanceRequest(task=task, payload=jsonable(payload))
        try:
            content = self._guidance.complete(request).content
        except _HOOK_EXCEPTIONS as exc:
            self._advisory[key] = {"error": str(exc)}
            return None
        self._advisory[key] = content
        return content

    def interpret(self, summary: dict) -> None:
        """``interpret_discovery`` over the summary WITHOUT the effects key."""
        self._call(
            "interpret_discovery",
            {"candidate": self._candidate.model_dump(), "summary": dict(summary),
             "context": None},
            "interpret_discovery",
        )

    def audit(self, validation: dict, summary: dict) -> None:
        """``audit_assumptions``; a truthy veto sets FLAGS only (never gates)."""
        content = self._call(
            "audit_assumptions",
            {"candidate": self._candidate.model_dump(), "validation": validation},
            "audit_assumptions",
        )
        if content is not None and content.get("veto"):
            self._advisory["vetoed"] = True
            summary["advisory_veto"] = True  # flag only; effects still computed

    def review_control(self, payload: dict) -> dict | None:
        """``review_control_group`` over the GESS extras; returns the content."""
        return self._call("review_control_group", payload, "control_review")


@dataclass
class ConfigRecord:
    """One enumerated design configuration and what happened to it (spec 6b)."""

    candidate: DesignCandidate
    source: str  # "plan" | "exhaustive"
    status: str  # "scanned" | "skipped_budget" | "failed" | "invalid"
    llr: float | None = None  # observed max LLR; None unless scanned — never 0.0
    p_value: float | None = None  # randomization-test p; None unless scanned
    n_discoveries: int = 0
    summary: dict = field(default_factory=dict)  # design-specific block; {} unless scanned
    advisory: dict = field(default_factory=dict)  # guidance hooks (spec 6c); ALWAYS advisory
    error: str | None = None
    # Row bookkeeping (issue #1): what listwise deletion did to THIS config's
    # dataset. None = the dataset was never built (invalid/skipped/rebuild
    # failure); {} row_loss = built with zero loss. Stamped before the runner,
    # so failed configs carry it too.
    n_rows_input: int | None = None
    n_rows_used: int | None = None
    row_loss: dict | None = None  # top per-column attributable losses (<= 3, desc)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.model_dump(),
            "source": self.source,
            "status": self.status,
            "llr": self.llr,
            "p_value": self.p_value,
            "n_discoveries": self.n_discoveries,
            "n_rows_input": self.n_rows_input,
            "n_rows_used": self.n_rows_used,
            "row_loss": self.row_loss,
            "summary": self.summary,
            "advisory": self.advisory,
            "error": self.error,
        }


@dataclass
class DiscoverReport:
    """All configs in execution order (plan-ranked first) + coverage block."""

    configs: list[ConfigRecord]
    searched: dict  # spec 6b coverage: counts, effective budget, candidate split
    best_index: int | None  # argmax llr over scanned configs; None if none scanned
    guidance_log_path: str | None

    def best(self) -> ConfigRecord | None:
        return None if self.best_index is None else self.configs[self.best_index]

    def to_json(self) -> str:
        payload = {
            "configs": [rec.to_dict() for rec in self.configs],
            "searched": self.searched,
            "best_index": self.best_index,
            "guidance_log_path": self.guidance_log_path,
        }
        return json.dumps(jsonable(payload), indent=1)

    def save(self, out: str | Path) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "discover_report.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def enumerate_configs(data: Dataset, design: str = "auto") -> list[DesignCandidate]:
    """Exhaustive design configurations derivable from the bound dataset spec.

    One rdd candidate (requires a nonempty forcing list) and one did candidate
    (requires ``spec.time``), filtered by ``design``.
    """
    if design not in _DESIGNS:
        raise ValueError(f"design must be one of {_DESIGNS}, got {design!r}")
    spec = data.spec
    out: list[DesignCandidate] = []
    if design in ("auto", "rdd") and spec.forcing:
        out.append(DesignCandidate(
            design="rdd", treatment=spec.treatment, outcome=spec.outcome,
            forcing=list(spec.forcing), rationale="bound dataset spec (exhaustive)",
        ))
    if design in ("auto", "did") and spec.time is not None:
        out.append(DesignCandidate(
            design="did", treatment=spec.treatment, outcome=spec.outcome, forcing=[],
            unit=spec.unit, time=spec.time, rationale="bound dataset spec (exhaustive)",
        ))
    return out


def _effective_budget(search_plan: SearchPlan | None, budget: dict | None) -> dict:
    """defaults <- search_plan.budget (hints) <- budget arg (explicit wins).

    ``search_plan.budget`` is agent-provided against a schema_hint declaring
    ``additionalProperties: true`` (Epoch dogfood finding 3): unknown keys
    (e.g. a ``note``) are advertised as acceptable, so they are ignored here
    rather than crashing the plan discover solicited. The explicit ``budget``
    argument is caller-typed and still raises on unknown keys.
    """
    eff = dict(_BUDGET_DEFAULTS)
    plan_budget = search_plan.budget if search_plan else {}
    eff.update({k: v for k, v in plan_budget.items() if k in _BUDGET_DEFAULTS})
    src = budget or {}
    unknown = sorted(set(src) - set(_BUDGET_DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown budget keys: {unknown}; known keys: {sorted(_BUDGET_DEFAULTS)}"
        )
    eff.update(src)
    return eff


def _candidate_error(candidate: DesignCandidate, df: pd.DataFrame) -> str | None:
    """Why the candidate cannot be scanned against ``df``, or None if it can."""
    cols = [candidate.treatment, *candidate.forcing]
    cols += [c for c in (candidate.outcome, candidate.unit, candidate.time) if c is not None]
    missing = sorted({c for c in cols if c not in df.columns})
    if missing:
        return f"columns not in dataframe: {missing}"
    bad = [c for c in candidate.forcing if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        return f"forcing columns must be numeric: {bad}"
    return None


def _dataset_for(data: Dataset, c: DesignCandidate, known_outcomes: set[str]) -> Dataset:
    """The bound Dataset when the candidate equals its spec, else a rebuilt one.

    The rebuilt spec mirrors ``Dataset.from_csv`` defaults: covariates = all
    columns minus {treatment, outcome} — and minus every OTHER candidate's
    outcome (issue #7): a foreign outcome in the covariates feeds the
    background treatment model and its NaNs listwise-delete scan rows. A
    column the candidate itself uses as forcing/time/unit keeps that role.
    The bound dataset is repaired the same way when its ``spec.covariates``
    smuggle a foreign outcome in; otherwise it passes through untouched.

    Both rebuilds start from ``data.df_input`` (issue #30): the bound Dataset
    already listwise-deleted on ITS covariates at construction, so rebuilding
    from ``data.df`` would keep rows deleted for a foreign outcome the new
    spec no longer declares — truncating the scan sample and defeating the
    issue-#1 row bookkeeping (``n_rows_input`` counted the truncated frame).
    """
    spec = data.spec
    same_roles = c.treatment == spec.treatment and c.outcome == spec.outcome
    if c.design == "rdd":
        bound = same_roles and sorted(c.forcing) == sorted(spec.forcing)
    else:
        bound = same_roles and c.time == spec.time and c.unit == spec.unit
    own_roles = {col for col in (*c.forcing, c.time, c.unit) if col is not None}
    foreign_outcomes = known_outcomes - own_roles - {c.outcome}
    if bound:
        leaked = foreign_outcomes & set(spec.covariates)
        if not leaked:
            return data
        kept = [col for col in spec.covariates if col not in leaked]
        return Dataset(data.df_input, spec.model_copy(update={"covariates": kept}))
    reserved = {c.treatment} | ({c.outcome} if c.outcome else set()) | foreign_outcomes
    covariates = [col for col in data.df_input.columns if col not in reserved]
    if c.design == "rdd":
        new_spec = DatasetSpec(treatment=c.treatment, outcome=c.outcome,
                               forcing=list(c.forcing), covariates=covariates)
    else:
        new_spec = DatasetSpec(treatment=c.treatment, outcome=c.outcome, forcing=[],
                               covariates=covariates, time=c.time, unit=c.unit)
    return Dataset(data.df_input, new_spec)


def _run_rdd(ds: Dataset, budget: dict, rng: np.random.Generator,
             hooks: _GuidanceHooks) -> tuple:
    """LoRD3 scan + randomization/placebo/density + local 2SLS effects."""
    k, q, degree = int(budget["k"]), int(budget["q"]), int(budget["degree"])
    coarse_block, geometry, search = None, None, None
    if budget["coarse"]:
        # Geometry depends only on Z_std (identical across null replicas):
        # build once, share with both stages and the calibration below.
        geometry = build_geometry(ds.Z_std, k)
        ctf = coarse_to_fine_scan(ds, k=k, n_coarse=int(budget["n_coarse"]),
                                  degree=degree, rng=rng, geometry=geometry)
        res = ctf.result  # fine-stage, full-resolution discoveries
        coarse_block = {"frac_centers_scanned": ctf.frac_centers_scanned, **ctf.params}
        # Issue #21: the observed statistic is a coarse-to-fine max, so each
        # null replica reruns the same coarse-to-fine search on its own T*
        # (frozen treatment-independent coarse subsample, replica's own
        # localization) — a full-scan replica max is stochastically larger
        # and would inflate the p-value.
        search = coarse_to_fine_search(
            ds, ctf.coarse_result.centers, k=k, top_m=int(ctf.params["top_m"]),
            radius_mult=float(ctf.params["radius_mult"]), model=res.model,
            degree=degree, geometry=geometry,
        )
    else:
        res = lord3_scan(ds, k=k, degree=degree, rng=rng)
    if not res.discoveries:
        raise ValueError("no scoreable neighborhood")
    rand = randomization_test(ds, res, Q=q, rng=rng, scan_kwargs={"k": k, "degree": degree},
                              geometry=geometry, search=search)
    top = res.discoveries[0]
    placebo = placebo_tests(ds, top)
    dens = density_test(ds, top)
    summary = {
        "design": "rdd",
        "center_z": ds.Z[top.center_index].tolist(),
        "normal": top.normal.tolist(),
        "forcing_influence": dict(
            zip(ds.spec.forcing, np.abs(top.normal).tolist(), strict=True)
        ),
        "llr": top.llr,
        "p_value": rand.p_value,
        # Issue #34: passed is None (JSON null) for a vacuous battery, with
        # the note saying why — never a bare True a consumer could read as a
        # real pass.
        "placebo_passed": placebo.passed,
        "placebo_note": placebo.note,
        "placebo_holm": placebo.p_holm,
        "density_p": dens.p_value,
        "density_se": dens.se,
        "coarse": coarse_block,
    }
    # Advisory hooks after validation, before effects (spec 6c); the audit
    # fires even when outcome is None — it is about the design, and a veto is
    # a flag only: effects below are computed regardless.
    hooks.interpret(summary)
    hooks.audit({"p_value": rand.p_value, "placebo_passed": placebo.passed,
                 "placebo_note": placebo.note, "placebo_holm": placebo.p_holm,
                 "density_p": dens.p_value, "density_se": dens.se}, summary)
    effects: dict = {}
    if ds.y is not None:
        for est in (local_2sls(ds, top), wald_estimate(ds, top)):
            effects[est.method] = {
                "tau": est.tau, "se": est.se, "ci": list(est.ci),
                "first_stage_t": est.first_stage_t,
                "weak_instrument": est.weak_instrument,
            }
    summary["effects"] = effects
    return float(rand.observed_max_llr), float(rand.p_value), len(res.discoveries), summary


def _run_did(ds: Dataset, budget: dict, rng: np.random.Generator,
             hooks: _GuidanceHooks) -> tuple:
    """SuDDDS scan + panel randomization/composition/anticipation + effects."""
    q, degree, bins = int(budget["q"]), int(budget["degree"]), int(budget["bins"])
    windows = budget["windows"]
    if windows is not None:
        windows = tuple(float(w) for w in windows)
    # audit 19's Bernoulli auto-matching conflicts with single_delta's
    # Gaussian profile GLR on binary treatments: resolve the DEFAULT
    # combination to the thesis-parity normal model instead of failing
    # every binary-treatment did config (dogfood finding). An explicit
    # model='bernoulli' still raises inside suddds_scan.
    method = budget["method"]
    model = resolve_default_model(budget["model"], method)
    panel = build_panel(ds, bins=bins)
    res = suddds_scan(ds, windows=windows, restarts=int(budget["restarts"]),
                      model=model, method=method,
                      bins=bins, degree=degree, rng=rng, panel=panel)
    if not res.discoveries:
        raise ValueError("no qualifying discovery")
    rand = panel_randomization_test(ds, res, Q=q, rng=rng,
                                    scan_kwargs={"bins": bins, "degree": degree})
    top = res.discoveries[0]
    comp = composition_test(panel, top)
    # anticipation_test refits its own nuisance on the pre-period sub-panel
    # (issue #12): a full-panel background would leak the real jump into the
    # trend coefficients and fail clean discoveries.
    antic = anticipation_test(panel, top, model=res.model, degree=degree)
    summary = {
        "design": "did",
        "subset_values": top.subset_values,
        "t0": top.t0,
        "window": top.window,
        "llr": top.llr,
        "p_value": rand.p_value,
        "null_kind": rand.null_kind,
        "composition_passed": comp.passed,
        "anticipation_passed": antic.passed,
        "searched_windows": list(res.windows),
        "restarts": res.restarts,
    }
    # Advisory hooks after validation, before effects (spec 6c) — see _run_rdd.
    hooks.interpret(summary)
    hooks.audit({"p_value": rand.p_value, "null_kind": rand.null_kind,
                 "composition_passed": comp.passed,
                 "anticipation_passed": antic.passed}, summary)
    effects: dict = {}
    if ds.y is not None:
        # gess control precomputed ONCE: review_control_group slots between
        # the fit and the reporting without a refit; τ̂/se/p are computed and
        # reported REGARDLESS of the review — a veto is only ever a flag.
        gess = gess_control(panel, top)
        review = hooks.review_control({
            "profile": gess.extras["profile"],
            "expansions": gess.extras["expansions"],
            "mse_trace": gess.extras["mse_trace"],
            "subset_values": top.subset_values,
            "n_control": gess.extras["n_control"],
            "n_tau": gess.extras["n_tau"],
        })
        controls: dict[str, object] = {"dd": "dd", "synthetic": "synthetic", "gess": gess}
        for name, control in controls.items():
            eff = did_effect(panel, top, control=control)
            tau_rand = tau_randomization_test(panel, top, control=name, rng=rng)
            effects[name] = {"tau": eff.tau, "se": eff.se, "p": tau_rand.p_value,
                             "pre_mse": eff.pre_mse, "dose": eff.dose}
        if review is not None:
            effects["gess"]["vetoed_by_guidance"] = bool(review.get("veto", False))
    summary["effects"] = effects
    return float(rand.observed_max_llr), float(rand.p_value), len(res.discoveries), summary


def discover(
    data: Dataset,
    design: str = "auto",
    guidance: GuidanceBackend | None = None,
    search_plan: SearchPlan | None = None,
    rng: np.random.Generator | None = None,
    budget: dict | None = None,
    out: str | Path | None = None,
) -> DiscoverReport:
    """Scan every enumerated configuration: plan-ranked first, exhaustive still.

    Effective budget = ``_BUDGET_DEFAULTS`` <- ``search_plan.budget`` (hints;
    unknown keys ignored per the open schema_hint) <- ``budget`` arg (explicit
    wins; unknown keys raise ValueError naming them). Plan candidates run in
    ranked order (invalid ones are recorded as
    ``status="invalid"``, not dropped), then the exhaustive remainder from
    :func:`enumerate_configs`, deduped on ``DesignCandidate.key()``. Once
    ``max_configs`` scan attempts have happened, the rest are listed with
    ``status="skipped_budget"`` — the report always states full coverage
    (spec 6b). A failing config is isolated (``status="failed"``, llr/p stay
    None) and never kills the sweep.

    With ``guidance``, three ADVISORY hooks fire per scanned config, in this
    order (the contract MockBackend response lists rely on):
    ``interpret_discovery`` (candidate + summary WITHOUT effects) ->
    ``audit_assumptions`` (candidate + validation block; a truthy ``veto``
    sets ``advisory["vetoed"]`` and ``summary["advisory_veto"]`` — flags
    only) -> on the did GESS path ``review_control_group`` (over the
    ``gess_control`` extras; a veto sets
    ``effects["gess"]["vetoed_by_guidance"]``). Statistics are NEVER gated or
    altered and hooks consume no rng: stripping the advisory keys from the
    report JSON yields output identical to a ``guidance=None`` run. A hook
    failure (``TimeoutError``/``ValueError``/``RuntimeError``) is recorded as
    ``advisory[<hook>] = {"error": ...}`` and the config stays ``scanned``.
    Payloads carry only summary statistics — never raw outcome values. When
    ``out`` is given the backend is wrapped in :class:`LoggedBackend`
    (exactly as ``study()`` does): one ``guidance_log.jsonl`` line per hook
    call.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    if design not in _DESIGNS:
        raise ValueError(f"design must be one of {_DESIGNS}, got {design!r}")
    eff_budget = _effective_budget(search_plan, budget)

    guidance_log_path: str | None = None
    if out is not None and guidance is not None:
        # No backend => no hook will ever log: recording a path to a file that
        # is never created misleads downstream readers (dogfood finding).
        log = GuidanceLog(Path(out) / "guidance_log.jsonl")
        guidance_log_path = str(log.path)
        guidance = LoggedBackend(guidance, log)  # one JSONL line per hook call

    # -- config list (spec 6b: plan orders, never truncates) ------------------
    # Dedup identity is (scan key, outcome) — issue #31: DesignCandidate.key()
    # stays outcome-blind because the SCAN never reads the outcome, but an
    # execution record is outcome-specific (effects are estimated against
    # ``c.outcome``). Keying on the blind key alone let a plan candidate for
    # y2 (or with no outcome at all) absorb the bound dataset's y record, so
    # y was never estimated — a plan must order the search, never truncate it.
    records: list[ConfigRecord] = []
    seen: set[tuple] = set()
    n_plan = 0
    if search_plan is not None:
        for c in search_plan.ranked():
            if design != "auto" and c.design != design:
                continue
            n_plan += 1
            seen.add((c.key(), c.outcome))
            err = _candidate_error(c, data.df)
            records.append(ConfigRecord(
                candidate=c, source="plan",
                status="invalid" if err else "pending", error=err,
            ))
    n_exhaustive = 0
    for c in enumerate_configs(data, design):
        if (c.key(), c.outcome) in seen:
            continue  # absorbed by an identical plan candidate (same outcome)
        n_exhaustive += 1
        seen.add((c.key(), c.outcome))
        err = _candidate_error(c, data.df)
        records.append(ConfigRecord(
            candidate=c, source="exhaustive",
            status="invalid" if err else "pending", error=err,
        ))

    # Issue #7: every candidate's outcome (plus the bound spec's) is reserved
    # from every scan's covariates — an outcome named by ONE candidate must
    # never feed another candidate's background model or listwise deletion.
    known_outcomes = {r.candidate.outcome for r in records if r.candidate.outcome is not None}
    if data.spec.outcome is not None:
        known_outcomes.add(data.spec.outcome)

    # -- sequential execution within budget ------------------------------------
    max_configs = eff_budget["max_configs"]
    n_attempted = 0
    for rec in records:
        if rec.status == "invalid":
            continue
        if max_configs is not None and n_attempted >= int(max_configs):
            rec.status = "skipped_budget"  # still listed, spec 6b
            continue
        n_attempted += 1
        hooks = _GuidanceHooks(guidance, rec.candidate, rec.advisory)
        try:
            ds = _dataset_for(data, rec.candidate, known_outcomes)
            # Stamped BEFORE the runner (issue #1): a config that fails
            # downstream still records what listwise deletion did to its rows
            # — failed designs need the bookkeeping most.
            rec.n_rows_input = ds.n_rows_input
            rec.n_rows_used = ds.n_rows_used
            rec.row_loss = ds.top_row_loss()
            runner = _run_rdd if rec.candidate.design == "rdd" else _run_did
            llr, p_value, n_disc, summary = runner(ds, eff_budget, rng, hooks)
        except _CONFIG_EXCEPTIONS as exc:
            rec.status = "failed"
            rec.error = str(exc)  # llr/p stay None — never fabricated
            continue
        rec.status = "scanned"
        rec.llr = llr
        rec.p_value = p_value
        rec.n_discoveries = n_disc
        rec.summary = summary

    searched = {
        "n_total": len(records),
        "n_scanned": sum(r.status == "scanned" for r in records),
        "n_skipped_budget": sum(r.status == "skipped_budget" for r in records),
        "n_failed": sum(r.status == "failed" for r in records),
        "n_invalid": sum(r.status == "invalid" for r in records),
        "budget": eff_budget,
        "plan_candidates": n_plan,
        "exhaustive_candidates": n_exhaustive,
    }

    scanned_idx = [i for i, r in enumerate(records) if r.status == "scanned"]
    best_index = None
    if scanned_idx:
        def _llr_key(i: int) -> float:
            v = records[i].llr
            return v if v is not None and np.isfinite(v) else float("-inf")

        best_index = max(scanned_idx, key=_llr_key)

    report = DiscoverReport(
        configs=records,
        searched=searched,
        best_index=best_index,
        guidance_log_path=guidance_log_path,
    )
    if out is not None:
        report.save(out)
    return report
