"""natex command-line interface."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from natex.data.registry import REGISTRY, data_root, verify
from natex.data.spec import Dataset
from natex.dee.debias import dee_debias
from natex.dee.vknn import select_m_prime
from natex.did.effects import did_effect, tau_randomization_test
from natex.did.panel import build_panel
from natex.did.suddds import resolve_default_model, suddds_scan
from natex.discover import discover as run_discover
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.intake.analyst import IntakeReport
from natex.intake.analyst import study as run_study
from natex.intake.prep import PrepPlan
from natex.iv.donors import sc_placebo_test, select_donors, unit_time_matrix
from natex.iv.pipeline import discover_instruments
from natex.jsonutil import jsonable
from natex.kink import difference_in_kinks, regression_kink
from natex.llm import AgentBackend, AnthropicBackend, GeminiBackend, GuidanceBackend
from natex.rdd.lord3 import lord3_scan
from natex.report.bundle import ResultsBundle
from natex.report.paper import render_paper
from natex.report.research_brief import research_brief
from natex.scan.coarse import coarse_to_fine_scan, coarse_to_fine_search
from natex.scan.geometry import build_geometry
from natex.survey import survey as run_survey
from natex.validate.density import density_test
from natex.validate.panel import (
    anticipation_test,
    composition_test,
    panel_randomization_test,
)
from natex.validate.placebo import placebo_tests
from natex.validate.randomization import randomization_test

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Automated natural-experiment discovery."""


_clean = jsonable  # extracted to natex.jsonutil; alias kept for the commands below


@app.command()
def datasets(
    root: Path = typer.Option(None, help="benchmark data root; default: env NATEX_DATA"),
):
    """Registry status: one line per dataset — found/missing, rows, ok, fetch source.

    Informational only: always exits 0. Missing entries print the registry's
    fetch-instruction string so the local data root can be reconstructed.
    """
    for name, info in REGISTRY.items():
        st = verify(name, root=root)
        if st.found:
            typer.echo(f"{name}  found  rows={st.n_rows}  ok={st.ok}  path={st.path}")
        else:
            typer.echo(f"{name}  missing  rows=?  ok=False  fetch: {info.source}")


@app.command("fetch-data")
def fetch_data(
    name: str,
    root: Path = typer.Option(None, help="data root; default env NATEX_DATA"),
    force: bool = typer.Option(False, "--force", help="re-download over an existing file"),
):
    """Download a dataset that has a public direct URL into the data root.

    Login-gated datasets print their fetch instructions and exit(1). The
    download streams to a temp file next to the target and is renamed
    atomically; afterwards ``verify(name)`` runs and its result is reported.
    """
    info = REGISTRY.get(name)
    if info is None:
        typer.echo(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
        raise typer.Exit(code=1)
    if info.fetch_url is None:
        typer.echo(f"{name} has no public direct download (login-gated).")
        typer.echo(f"How to obtain it: {info.source}")
        raise typer.Exit(code=1)
    try:
        base = data_root(root)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    dest = base / info.relpath
    if dest.exists() and not force:
        typer.echo(f"{dest} already exists; pass --force to re-download.")
        raise typer.Exit(code=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out_f, urllib.request.urlopen(info.fetch_url) as resp:
            shutil.copyfileobj(resp, out_f)
        os.replace(tmp_name, dest)  # atomic within the same directory
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    typer.echo(f"downloaded {info.fetch_url} -> {dest}")
    st = verify(name, root=base)
    typer.echo(f"verify: rows={st.n_rows}  ok={st.ok}  {st.message}")
    if not st.ok:
        raise typer.Exit(code=1)


_BACKEND_HELP = (
    "null|agent|anthropic|gemini. 'null' (default) needs no network or API key; "
    "anthropic/gemini require: pip install 'natex-discovery[llm]'"
)


def _make_backend(
    backend: str, model: str | None, workdir: Path | None
) -> GuidanceBackend | None:
    """Guidance backend from the CLI flags, or None for the no-LLM default.

    ``"null"`` returns None: ``study()``/``discover()`` substitute NullBackend
    internally where needed (``natex study`` ALWAYS uses at least NullBackend
    heuristics), and passing None keeps ``discover`` hook-free for the null
    case. A missing ``[llm]`` extra exits 2 with the install message, no
    traceback; an unknown backend name exits 2 naming it.
    """
    try:
        if backend == "null":
            return None
        if backend == "agent":
            return AgentBackend(workdir)
        if backend == "anthropic":
            return AnthropicBackend(**({"model": model} if model else {}))
        if backend == "gemini":
            return GeminiBackend(**({"model": model} if model else {}))
    except ImportError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    typer.echo(f"--backend must be one of null|agent|anthropic|gemini, got {backend!r}")
    raise typer.Exit(code=2)


@app.command()
def study(
    csv: Path,
    context: str = typer.Option(None, help="free-text dataset context passed to the analyst"),
    backend: str = typer.Option("null", help=_BACKEND_HELP),
    model: str = typer.Option(None, help="LLM model name (--backend anthropic|gemini)"),
    workdir: Path = typer.Option(
        None, help="agent-backend request/response dir; default OUT/guidance"
    ),
    seed: int = typer.Option(0, help="RNG seed (converted once to the run's single numpy Generator)"),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """Stage-0 analyst pass: profile -> understand -> prep plan -> search plan.

    Writes ``out/intake_report.json`` (feed it to ``natex discover --plan``),
    ``out/prep_plan.json`` (the declarative prep plan alone, user-editable —
    feed an edited copy back via ``natex discover --plan ... --prep-plan``) and
    ``out/guidance_log.jsonl`` (every guidance request+response). Works fully
    offline with the default ``--backend null`` — study always applies at
    least the deterministic NullBackend heuristics.
    """
    guidance = _make_backend(
        backend, model, workdir if workdir is not None else out / "guidance"
    )
    report = run_study(
        csv, context=context, guidance=guidance, rng=np.random.default_rng(seed), out=out
    )
    report_path = report.save(out)
    und = report.understanding
    typer.echo(f"shape: {und.shape}  unit of observation: {und.unit_of_observation}")
    ranked = report.search_plan.ranked()
    if ranked:
        typer.echo(f"candidates: {len(ranked)}  top: {_candidate_line(ranked[0])}")
    else:
        typer.echo("candidates: 0")
    for err in report.guidance_errors:
        typer.echo(f"warning: {err}")
    typer.echo(f"report: {report_path}")
    typer.echo(f"prep plan: {out / 'prep_plan.json'}")
    typer.echo(f"guidance log: {out / 'guidance_log.jsonl'}")


def _candidate_line(c) -> str:
    """One-line design/treatment/forcing-or-time rendering of a DesignCandidate."""
    where = "forcing=" + ",".join(c.forcing) if c.design == "rdd" else f"time={c.time}"
    return f"{c.design} treatment={c.treatment} {where}"


def _given(ctx: typer.Context, name: str) -> bool:
    """True when the option was explicitly passed on the command line.

    Compared by enum NAME: typer >= 0.16 vendors click, so the
    click.core.ParameterSource class itself is not importable.
    """
    src = ctx.get_parameter_source(name)
    return src is not None and src.name == "COMMANDLINE"


def _parse_col_value_pairs(items: list[str] | None, option: str) -> dict[str, float]:
    """Repeatable ``COL=VALUE`` flags -> ``{col: float(value)}``.

    A malformed item (no ``=``, empty column, non-numeric value) exits 2
    naming the offending item, no traceback.
    """
    pairs: dict[str, float] = {}
    for item in items or []:
        col, sep, raw = item.partition("=")
        try:
            if not col or not sep:
                raise ValueError(item)
            pairs[col] = float(raw)
        except ValueError:
            typer.echo(f"{option} expects COL=VALUE with a numeric VALUE, got {item!r}")
            raise typer.Exit(code=2) from None
    return pairs


@app.command()
def survey(
    ctx: typer.Context,
    csv: Path,
    context: str = typer.Option(None, help="free-text dataset context passed to the analyst"),
    backend: str = typer.Option("null", help=_BACKEND_HELP),
    model: str = typer.Option(None, help="LLM model name (--backend anthropic|gemini)"),
    workdir: Path = typer.Option(
        None, help="agent-backend request/response dir; default OUT/guidance"
    ),
    out: Path = typer.Option(Path("out/survey"), help="output directory"),
    seed: int = typer.Option(0, help="RNG seed (converted once to the run's single numpy Generator)"),
    time: str = typer.Option(None, help="panel time column (did/sc families)"),
    unit: str = typer.Option(None, help="panel unit column (did/sc families)"),
    cutoff: list[str] = typer.Option(
        None, "--cutoff", help="declared kink cutoff COL=VALUE (repeatable)"
    ),
    instrument: list[str] = typer.Option(
        None, "--instrument", help="candidate instrument column (repeatable)"
    ),
    threshold: list[str] = typer.Option(
        None, "--threshold", help="declared bunching threshold COL=VALUE (repeatable)"
    ),
    k: int = typer.Option(50, help="scan neighborhood size (forwarded into the budget when passed)"),
    q: int = typer.Option(99, help="randomization replicas (forwarded into the budget when passed)"),
    coarse: bool = typer.Option(False, "--coarse/--no-coarse",
                                help="coarse-to-fine scan (large datasets)"),
    n_coarse: int = typer.Option(2000, help="coarse-stage center subsample size"),
    max_configs: int = typer.Option(
        None, "--max-configs",
        help="scan-attempt budget across configurations; the remainder is "
             "listed as skipped_budget, never dropped",
    ),
):
    """One-command systematic design survey: run the dataset against ALL seven
    method families (rdd, did, kink, iv, sc, bunching, dee) and write one
    visual report with an applicability verdict per family.

    Writes OUT/survey.json, OUT/report.md (always) and OUT/report.html (with
    the [report] extra), per-family details under OUT/families/ and the
    stage-0 analyst artifacts under OUT/intake/. Failed families are a
    recorded outcome, not a CLI failure — exit 0 whenever survey.json was
    written.
    """
    cutoffs = _parse_col_value_pairs(cutoff, "--cutoff")
    thresholds = _parse_col_value_pairs(threshold, "--threshold")
    try:
        pd.read_csv(csv)  # readability check only; survey() re-reads and records the path
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        # ValueError covers pandas ParserError/EmptyDataError
        typer.echo(f"could not read {csv}: {exc}")
        raise typer.Exit(code=2) from None
    guidance = _make_backend(
        backend, model, workdir if workdir is not None else out / "guidance"
    )
    budget: dict = {}
    for key, value in (("k", k), ("q", q), ("coarse", coarse), ("n_coarse", n_coarse)):
        if _given(ctx, key):
            budget[key] = value
    if max_configs is not None:
        budget["max_configs"] = max_configs
    res = run_survey(
        csv, context=context, guidance=guidance, rng=np.random.default_rng(seed),
        out_dir=out, budget=budget or None, time=time, unit=unit,
        cutoffs=cutoffs or None, instruments=list(instrument or []) or None,
        thresholds=thresholds or None, seed=seed,
    )
    for name, fam in res.families.items():
        typer.echo(f"{name:<9}{fam.status:<12}{fam.reason[:70]}")
    for note in res.coverage.get("notes", []):
        typer.echo(f"note: {note}")
    typer.echo(f"report: {out / (res.report_html or res.report_md)}")
    typer.echo(f"survey: {out / 'survey.json'}")


def _discover_plan(
    ctx: typer.Context,
    *,
    csv: Path | None,
    plan: Path,
    prep_plan: Path | None,
    backend: str,
    model: str | None,
    workdir: Path | None,
    max_configs: int | None,
    design: str,
    k: int,
    q: int,
    coarse: bool,
    n_coarse: int,
    seed: int,
    out: Path,
) -> None:
    """Plan-driven branch of ``discover`` (spec 6b through the CLI).

    Loads the ``natex study`` IntakeReport, re-applies its prep plan to the
    csv (or the report's recorded source), and runs :func:`natex.discover`
    with the report's search plan: ranked candidates first, exhaustive
    remainder still, every budget cut listed as ``skipped_budget``. Budget =
    plan hints overridden by CLI options the user explicitly passed
    (k/q/coarse/n-coarse) plus ``--max-configs``. Exits 1 when no
    configuration scanned successfully.
    """

    def given(name: str) -> bool:
        return _given(ctx, name)

    plan_design = design if given("design") else "auto"
    if plan_design not in ("auto", "rdd", "did"):
        typer.echo(f"--design must be 'auto', 'rdd' or 'did' with --plan, got {design!r}")
        raise typer.Exit(code=2)
    try:
        report = IntakeReport.load(plan)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"could not load --plan {plan}: {exc}")
        raise typer.Exit(code=2) from None
    if prep_plan is not None:
        # Issue #18: out/prep_plan.json is documented as user-editable, so an
        # edited copy must be loadable back — and never silently: the override
        # is echoed and recorded in guidance_errors (carried into the results
        # bundle's intake block). Replaced BEFORE prepare(), so the plan is
        # still validated against the real frame by PrepPlan.apply.
        try:
            content = json.loads(Path(prep_plan).read_text(encoding="utf-8"))
            report.prep_plan = PrepPlan.model_validate(content)
        except (OSError, ValueError) as exc:
            # ValueError covers json.JSONDecodeError and pydantic.ValidationError
            typer.echo(f"could not load --prep-plan {prep_plan}: {exc}")
            raise typer.Exit(code=2) from None
        typer.echo(f"prep plan overridden: {prep_plan}")
        report.guidance_errors.append(
            f"prep plan overridden from {prep_plan} (CLI --prep-plan)"
        )
    df = pd.read_csv(csv) if csv is not None else None
    try:
        ds = report.prepare(df=df)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    guidance = _make_backend(
        backend,
        model if given("model") else None,  # bare --model default is the did scan model
        workdir if workdir is not None else out / "guidance",
    )
    budget: dict = {}
    for key, value in (("k", k), ("q", q), ("coarse", coarse), ("n_coarse", n_coarse)):
        if given(key):
            budget[key] = value
    if max_configs is not None:
        budget["max_configs"] = max_configs
    rep = run_discover(
        ds, design=plan_design, guidance=guidance, search_plan=report.search_plan,
        rng=np.random.default_rng(seed), budget=budget, out=out,
    )
    path = rep.save(out)
    # Issue #2: also save the full ResultsBundle — seed, natex version, the
    # prepared dataset's data block and the intake provenance — so paper/brief
    # stop rendering "seed —" / "No dataset metadata was recorded".
    # discover_report.json above stays: it is a documented output and the
    # compat load path for pre-bundle directories.
    bundle_path = ResultsBundle.from_discover(
        rep, out, dataset=ds, intake=report, seed=seed
    ).save()
    s = rep.searched
    typer.echo(
        f"scanned {s['n_scanned']}/{s['n_total']} configs "
        f"({s['n_skipped_budget']} skipped by budget)"
    )
    best = rep.best()
    if best is None:
        typer.echo(f"no configuration scanned successfully; report: {path}")
        raise typer.Exit(code=1)
    typer.echo(
        f"best: {_candidate_line(best.candidate)}  "
        f"llr={best.llr:.2f}  p={best.p_value:.3f}"
    )
    typer.echo(f"report: {path}")
    typer.echo(f"results: {bundle_path}")


@app.command()
def discover(
    ctx: typer.Context,
    csv: Path = typer.Argument(None),
    treatment: str = typer.Option(None, help="treatment column (required without --plan)"),
    outcome: str = typer.Option(
        None, help="outcome column (discovery never reads it; needed for effect estimates)"
    ),
    forcing: str = typer.Option(None, help="comma-separated; default: all numeric"),
    k: int = typer.Option(50, help="scan neighborhood size"),
    q: int = typer.Option(99, help="randomization replicas"),
    seed: int = typer.Option(0, help="RNG seed (converted once to the run's single numpy Generator)"),
    degree: int = typer.Option(1, help="background polynomial degree of the treatment model"),
    coarse: bool = typer.Option(False, "--coarse/--no-coarse",
                                help="coarse-to-fine scan (large datasets)"),
    n_coarse: int = typer.Option(2000, help="coarse-stage center subsample size"),
    design: str = typer.Option("rdd", help="rdd (LoRD3 scan) | did (SuDDDS panel scan)"),
    time: str = typer.Option(None, help="panel time column (required for --design did)"),
    unit: str = typer.Option(None, help="panel unit column (optional; --design did)"),
    bins: int = typer.Option(4, help="quantile bins per numeric covariate (--design did)"),
    windows: str = typer.Option(
        None, help="comma-separated window widths, e.g. '8,10'; default: data-driven (did)"
    ),
    restarts: int = typer.Option(8, help="scan restarts per window (--design did)"),
    method: str = typer.Option(
        "single_delta", help="single_delta|wcc|greedy (--design did)"
    ),
    model: str = typer.Option(
        "auto", help="auto|normal|bernoulli (--design did; audit-19 model matching); "
                     "with --plan: LLM model name for --backend anthropic|gemini"
    ),
    plan: Path = typer.Option(
        None, help="intake_report.json from `natex study`: plan candidates scan first, "
                   "the exhaustive remainder still runs within budget (spec 6b)"
    ),
    prep_plan: Path = typer.Option(
        None, "--prep-plan",
        help="prep_plan.json overriding the plan's embedded prep plan, e.g. an "
             "edited `natex study` out/prep_plan.json (--plan mode only); the "
             "override is echoed and recorded in the bundle's intake provenance",
    ),
    backend: str = typer.Option("null", help=f"{_BACKEND_HELP} (--plan mode)"),
    workdir: Path = typer.Option(
        None, help="agent-backend request/response dir (--plan mode); default OUT/guidance"
    ),
    max_configs: int = typer.Option(
        None, "--max-configs",
        help="scan-attempt budget across configurations (--plan mode); the "
             "remainder is listed as skipped_budget, never dropped",
    ),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """Scan for natural experiments: LoRD3 RDD scan or SuDDDS DiD scan, with
    the validation battery and honest effect estimates.

    Requires CSV + --treatment, or --plan intake_report.json from `natex study`
    (plan candidates scan first, the exhaustive remainder runs within budget).
    Writes OUT/results.json; plan mode also writes OUT/discover_report.json.
    """
    if plan is not None:
        _discover_plan(
            ctx, csv=csv, plan=plan, prep_plan=prep_plan, backend=backend,
            model=model, workdir=workdir, max_configs=max_configs, design=design,
            k=k, q=q, coarse=coarse, n_coarse=n_coarse, seed=seed, out=out,
        )
        return
    if prep_plan is not None:
        typer.echo(
            "--prep-plan requires --plan intake_report.json "
            "(it overrides the plan's embedded prep plan)"
        )
        raise typer.Exit(code=2)
    if csv is None or treatment is None:
        typer.echo(
            "natex discover requires CSV and --treatment COLUMN "
            "(or --plan intake_report.json from `natex study`)"
        )
        raise typer.Exit(code=2)
    if design not in ("rdd", "did"):
        typer.echo(f"--design must be 'rdd' or 'did', got {design!r}")
        raise typer.Exit(code=2)
    if design == "did":
        _discover_did(
            csv, treatment=treatment, outcome=outcome, forcing=forcing, q=q,
            seed=seed, degree=degree, time=time, unit=unit, bins=bins,
            windows=windows, restarts=restarts, method=method, model=model, out=out,
        )
        return
    ds = Dataset.from_csv(
        csv, treatment=treatment, outcome=outcome,
        forcing=forcing.split(",") if forcing else None,
    )
    rng = np.random.default_rng(seed)
    coarse_block, geometry, search = None, None, None
    if coarse:
        geometry = build_geometry(ds.Z_std, k)
        ctf = coarse_to_fine_scan(ds, k=k, n_coarse=n_coarse, degree=degree, rng=rng,
                                  geometry=geometry)
        # Validation/estimation below operate on the fine-stage (full-resolution)
        # result; the coverage block reports what was and wasn't searched (spec 6b).
        res = ctf.result
        coarse_block = {"frac_centers_scanned": ctf.frac_centers_scanned, **ctf.params}
        # Issue #21: calibrate the coarse-to-fine observed statistic with
        # coarse-to-fine replicas (frozen coarse subsample, per-replica
        # localization); full-scan replica maxima would inflate the p-value.
        search = coarse_to_fine_search(
            ds, ctf.coarse_result.centers, k=k, top_m=int(ctf.params["top_m"]),
            radius_mult=float(ctf.params["radius_mult"]), model=res.model,
            degree=degree, geometry=geometry,
        )
    else:
        res = lord3_scan(ds, k=k, degree=degree, rng=rng)
    if not res.discoveries:
        # Issue #28: the audit-item-21 fast path skips every treatment-
        # homogeneous neighborhood, so a well-separated dataset can yield an
        # empty scan — exit cleanly (mirrors the did branch) instead of
        # tracebacking inside randomization_test.
        typer.echo(
            "no scoreable neighborhood: every size-k neighborhood is "
            "treatment-homogeneous; try a larger --k or check the forcing columns"
        )
        raise typer.Exit(code=1)
    rand = randomization_test(ds, res, Q=q, rng=rng, scan_kwargs={"k": k, "degree": degree},
                              geometry=geometry, search=search)
    top = res.discoveries[0]
    placebo = placebo_tests(ds, top)
    dens = density_test(ds, top)
    effects = {}
    if ds.y is not None:
        for est in (local_2sls(ds, top), wald_estimate(ds, top)):
            effects[est.method] = {
                "tau": est.tau, "se": est.se, "ci": list(est.ci),
                "first_stage_t": est.first_stage_t, "weak_instrument": est.weak_instrument,
            }
    payload = _clean(
        {
            # Issue #29: record the run's roles (the RESOLVED spec forcing, so
            # the default all-numeric list is persisted) — paper/brief read them.
            "params": {"treatment": treatment, "outcome": outcome,
                       "forcing": list(ds.spec.forcing),
                       "k": k, "q": q, "seed": seed, "degree": degree,
                       "coarse": coarse, "n_coarse": n_coarse, "csv": str(csv)},
            "scan": {"model": res.model, "p_value": rand.p_value,
                     "observed_max_llr": rand.observed_max_llr},
            "discoveries": [
                {
                    "center_z": ds.Z[d.center_index].tolist(),
                    "llr": d.llr,
                    "normal": d.normal.tolist(),
                    "forcing_influence": dict(
                        zip(ds.spec.forcing, np.abs(d.normal).tolist())
                    ),
                }
                for d in res.top(20)
            ],
            "validation": {
                "placebo_holm": placebo.p_holm, "placebo_passed": placebo.passed,
                "density_p": dens.p_value,
            },
            "effects": effects,
            **({"coarse": coarse_block} if coarse_block is not None else {}),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    typer.echo(f"model={res.model}  max LLR={rand.observed_max_llr:.2f}  scan p={rand.p_value:.3f}")
    if coarse_block is not None:
        typer.echo(
            f"coarse-to-fine: scanned {coarse_block['frac_centers_scanned']:.1%} of centers "
            f"(n_coarse={n_coarse}, top_m={coarse_block['top_m']})"
        )
    typer.echo(f"top center (raw z): {ds.Z[top.center_index]}")
    typer.echo(f"placebo passed: {placebo.passed}   density p: {dens.p_value:.3f}")
    if effects:
        e = effects["2sls"]
        typer.echo(f"2SLS tau={e['tau']:.3f} CI=({e['ci'][0]:.3f},{e['ci'][1]:.3f}) weak_iv={e['weak_instrument']}")
    typer.echo(f"results: {out / 'results.json'}")


@app.command()
def kink(
    csv: Path,
    design: str = typer.Option("rkd", help="rkd|dik (known-cutoff kink design)"),
    outcome: str = typer.Option(..., help="outcome column"),
    running: str = typer.Option(..., help="running-variable column"),
    treatment: str = typer.Option(
        None, help="observed policy/treatment column (fuzzy RKD or DiK)"
    ),
    policy_kink: float = typer.Option(
        None, "--policy-kink", help="known right-minus-left policy slope kink (sharp RKD)"
    ),
    policy_kink_change: float = typer.Option(
        None,
        "--policy-kink-change",
        help="known post-minus-pre change in the policy slope kink (sharp DiK)",
    ),
    time: str = typer.Option(None, help="time/period column (--design dik)"),
    t0: float = typer.Option(None, help="first post-policy period (--design dik)"),
    cutoff: float = typer.Option(0.0, help="known running-variable cutoff"),
    bandwidth: float = typer.Option(..., help="required symmetric bandwidth around cutoff"),
    degree: int = typer.Option(1, help="local polynomial degree (>=1)"),
    kernel: str = typer.Option("triangular", help="triangular|uniform|epanechnikov"),
    donut: float = typer.Option(0.0, help="exclude |running-cutoff| below this radius"),
    covariates: str = typer.Option(None, help="comma-separated numeric adjustment columns"),
    cluster: str = typer.Option(None, help="cluster column for CR1 covariance"),
    alpha: float = typer.Option(0.05, help="two-sided confidence-set size"),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """Estimate a known-cutoff sharp/fuzzy RKD or difference-in-kinks design.

    The slope convention is always right-minus-left. A fuzzy design estimates
    that slope contrast on ``--treatment``; a sharp design divides by the
    supplied policy kink. DiK forms post-minus-pre before taking the ratio.
    The command requires an explicit bandwidth because no automatic DiK
    bandwidth selector is established; inspect bandwidth, donut, and placebo
    cutoff sensitivity before treating the conventional interval as final.
    """
    if design not in ("rkd", "dik"):
        typer.echo(f"--design must be 'rkd' or 'dik', got {design!r}")
        raise typer.Exit(code=2)
    covariate_names = [c.strip() for c in covariates.split(",") if c.strip()] if covariates else []
    try:
        df = pd.read_csv(csv)
    except (OSError, pd.errors.ParserError) as exc:
        typer.echo(f"could not read {csv}: {exc!r}")
        raise typer.Exit(code=2) from None

    known: float | None
    if design == "rkd":
        if time is not None or t0 is not None:
            typer.echo("--time/--t0 apply only to --design dik")
            raise typer.Exit(code=2)
        if policy_kink_change is not None:
            typer.echo("--policy-kink-change applies only to --design dik")
            raise typer.Exit(code=2)
        known = policy_kink
    else:
        if time is None or t0 is None:
            typer.echo("--design dik requires both --time COLUMN and --t0 VALUE")
            raise typer.Exit(code=2)
        if policy_kink is not None:
            typer.echo("--policy-kink applies only to --design rkd")
            raise typer.Exit(code=2)
        known = policy_kink_change
    if (treatment is None) == (known is None):
        flag = "--policy-kink" if design == "rkd" else "--policy-kink-change"
        typer.echo(f"supply exactly one of --treatment (fuzzy) or {flag} (sharp)")
        raise typer.Exit(code=2)

    columns = [outcome, running, *covariate_names]
    columns += [c for c in (treatment, time, cluster) if c is not None]
    missing = sorted({c for c in columns if c not in df.columns})
    if missing:
        typer.echo(f"columns not in dataframe: {missing}")
        raise typer.Exit(code=2)

    def numeric_column(name: str) -> np.ndarray:
        try:
            return df[name].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"column {name!r} must be numeric") from None

    try:
        outcome_values = numeric_column(outcome)
        running_values = numeric_column(running)
        treatment_values = None if treatment is None else numeric_column(treatment)
        covariate_values = (
            np.column_stack([numeric_column(name) for name in covariate_names])
            if covariate_names
            else None
        )
        cluster_values = None if cluster is None else df[cluster].to_numpy()
        common = {
            "treatment": treatment_values,
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "degree": degree,
            "kernel": kernel,
            "donut": donut,
            "covariates": covariate_values,
            "clusters": cluster_values,
            "alpha": alpha,
        }
        if design == "rkd":
            estimate = regression_kink(
                outcome_values,
                running_values,
                policy_kink=policy_kink,
                **common,
            )
            post = None
        else:
            time_values = numeric_column(time)
            post = np.full(time_values.shape, np.nan, dtype=float)
            finite_time = np.isfinite(time_values)
            post[finite_time] = (time_values[finite_time] >= t0).astype(float)
            estimate = difference_in_kinks(
                outcome_values,
                running_values,
                post,
                policy_kink_change=policy_kink_change,
                **common,
            )
    except (TypeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None

    caveats = [
        "The cutoff and bandwidth are user-specified; report bandwidth, donut, and placebo-cutoff sensitivity (natex.kink: sensitivity_grid, placebo_kinks, covariate_kinks, event_study_kinks, density_kink_difference).",
        "The reported interval is conventional local-polynomial inference and may retain smoothing bias.",
    ]
    if design == "rkd":
        caveats.append(
            "Causal interpretation requires no kink in the non-policy outcome derivative and a continuous marginal response at the cutoff."
        )
    else:
        caveats.append(
            "Causal interpretation requires parallel changes in non-policy slope kinks and a time-stable marginal response at the cutoff."
        )
        if treatment is not None:
            caveats.extend(
                [
                    "Fuzzy DiK additionally requires stable latent policy-schedule composition at the cutoff (or valid reweighting).",
                    "Individual latent policy kink changes must have the same sign for positive-weight interpretation.",
                ]
            )
    payload = _clean(
        {
            "params": {
                "design": design,
                "outcome": outcome,
                "running": running,
                "treatment": treatment,
                "policy_kink": policy_kink,
                "policy_kink_change": policy_kink_change,
                "time": time,
                "t0": t0,
                "cutoff": cutoff,
                "bandwidth": bandwidth,
                "degree": degree,
                "kernel": kernel,
                "donut": donut,
                "covariates": covariate_names,
                "cluster": cluster,
                "alpha": alpha,
                "contrast": "right_minus_left",
                "csv": str(csv),
            },
            "estimate": asdict(estimate),
            "identification_caveats": caveats,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kink.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tau_text = "undefined" if not np.isfinite(estimate.tau) else f"{estimate.tau:.4f}"
    se_text = "undefined" if not np.isfinite(estimate.se) else f"{estimate.se:.4f}"
    fs_text = (
        "undefined" if not np.isfinite(estimate.first_stage_F) else f"{estimate.first_stage_F:.2f}"
    )
    typer.echo(
        f"{estimate.method}: tau={tau_text} se={se_text} "
        f"first-stage F={fs_text} weak={estimate.weak_first_stage}"
    )
    typer.echo(f"results: {path}")
    if not np.isfinite(estimate.tau):
        reason = estimate.extras.get(
            "reason", "the estimand is undefined because its denominator is zero or non-finite"
        )
        typer.echo(f"estimation failed: {reason}")
        raise typer.Exit(code=1)


@app.command()
def debias(
    csv: Path,
    treatment: str = typer.Option(..., help="treatment column"),
    outcome: str = typer.Option(None, help="outcome column (required for debiasing)"),
    forcing: str = typer.Option(None, help="comma-separated; default: all numeric"),
    k: int = typer.Option(50, help="scan neighborhood size"),
    degree: int = typer.Option(1, help="background polynomial degree of the treatment model"),
    m_prime: int = typer.Option(25, "--m-prime",
                                help="top-LLR candidates offered to the VKNN repair"),
    q_null: int = typer.Option(0, "--q-null",
                               help="> 0: select M' from a Q=q-null fitted-null Monte Carlo"),
    k_prime: int = typer.Option(250, "--k-prime", help="k'-NN ball size per experiment"),
    t_side: int = typer.Option(15, "--t-side", help="min support per hyperplane side"),
    grid: int = typer.Option(15, help="query lattice points per forcing dimension"),
    weighting: str = typer.Option("stacking", help="stacking|loo|mll"),
    seed: int = typer.Option(0, help="RNG seed (converted once to the run's single numpy Generator)"),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """DEE debiasing: scan -> VKNN repair -> local 2SLS -> GP debiasing.

    Runs the LoRD3 scan, repairs the top M' candidates into disjoint
    quasi-experiments, estimates local effects, cross-fits the observational
    T-learner (audit 9), fits the bias/direct GP surfaces, and writes
    ``out/dee_result.json`` with the model weights, the per-experiment effects
    table, raw/debiased/direct/mixture predictions (mean + sd) on a
    ``grid``-per-dimension query lattice spanning the observed forcing ranges,
    and the pipeline diagnostics. NaN values are serialized as null, never 0.
    """
    if outcome is None:
        typer.echo("debias requires --outcome COLUMN (the effect being debiased)")
        raise typer.Exit(code=2)
    ds = Dataset.from_csv(
        csv, treatment=treatment, outcome=outcome,
        forcing=forcing.split(",") if forcing else None,
    )
    rng = np.random.default_rng(seed)
    res_scan = lord3_scan(ds, k=k, degree=degree, rng=rng)
    if q_null > 0:
        rand = randomization_test(
            ds, res_scan, Q=q_null, rng=rng, scan_kwargs={"k": k, "degree": degree}
        )
        m_prime_used = select_m_prime(res_scan, rand.null_max_llrs)
    else:
        m_prime_used = int(m_prime)
    # query lattice: `grid` points per forcing dim spanning the observed range
    axes = [np.linspace(ds.Z[:, j].min(), ds.Z[:, j].max(), grid)
            for j in range(ds.Z.shape[1])]
    mesh = np.meshgrid(*axes, indexing="ij")
    query = np.column_stack([m.ravel() for m in mesh])
    res = dee_debias(
        ds, query, res_scan, m_prime=m_prime_used, k_prime=k_prime,
        t_side=t_side, weighting=weighting, rng=rng,
    )

    def _mean_sd(mean: np.ndarray, cov: np.ndarray | None) -> dict:
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0)) if cov is not None else np.full(
            mean.shape[0], np.nan
        )
        return {"mean": mean, "sd": sd}

    # model A cov = bias posterior cov; model B cov = direct posterior cov
    cov_a = res.mixture.post_a.cov if res.mixture is not None else None
    cov_b = res.mixture.post_b.cov if res.mixture is not None else None
    payload = _clean(
        {
            "params": {"k": k, "degree": degree, "m_prime": m_prime, "q_null": q_null,
                       "m_prime_used": m_prime_used, "k_prime": k_prime, "t_side": t_side,
                       "grid": grid, "weighting": weighting, "seed": seed, "csv": str(csv)},
            "weights": {"w_debias": res.weights.w_debias, "strategy": res.weights.strategy,
                        "detail": res.weights.detail},
            "experiments": [
                {
                    "center_z": ds.Z[e.center_index].tolist(),
                    "llr": e.llr,
                    "n_members": int(len(e.members)),
                    "tau": eff.tau,
                    "se": eff.se,
                    "first_stage_t": eff.first_stage_t,
                    "weak_instrument": eff.weak_instrument,
                    "used": used,
                    "obs_cate": obs,
                    "bias_obs": bias,
                    "noise_var": nv,
                }
                for e, eff, used, obs, bias, nv in zip(
                    res.vknn.experiments, res.effects, res.used.tolist(),
                    res.obs_at_centers, res.bias_obs, res.noise_var, strict=True,
                )
            ],
            "grid": {
                "query": res.query,
                "cate_raw": res.cate_raw,
                "cate_debiased": _mean_sd(res.cate_debiased, cov_a),
                "cate_direct": _mean_sd(res.cate_direct, cov_b),
                "mixture": _mean_sd(
                    res.mixture.mean if res.mixture is not None
                    else np.full(res.cate_raw.shape[0], np.nan),
                    res.mixture.cov if res.mixture is not None else None,
                ),
            },
            "diagnostics": res.diagnostics,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "dee_result.json").write_text(json.dumps(payload, indent=1))
    diag = res.diagnostics
    typer.echo(
        f"experiments: {diag['n_experiments']} repaired, "
        f"{diag['n_experiments_used']} used (m_prime={m_prime_used})"
    )
    if "reason" in diag:
        typer.echo(f"degenerate: {diag['reason']}")
    else:
        typer.echo(
            f"w_debias={res.weights.w_debias:.2f} ({res.weights.strategy})  "
            f"grid mean cate: raw={float(np.nanmean(res.cate_raw)):.3f} "
            f"debiased={float(np.nanmean(res.cate_debiased)):.3f} "
            f"direct={float(np.nanmean(res.cate_direct)):.3f}"
        )
    typer.echo(f"results: {out / 'dee_result.json'}")


@app.command()
def instruments(
    csv: Path,
    treatment: str = typer.Option(..., help="treatment column (the endogenous regressor)"),
    pool: str = typer.Option(
        None, help="comma-separated candidate instruments; "
                   "default: all numeric columns except treatment/outcome/controls"
    ),
    controls: str = typer.Option(None, help="comma-separated control columns"),
    outcome: str = typer.Option(
        None, help="outcome column; omit for selection only (discovery never reads the outcome)"
    ),
    honest: bool = typer.Option(
        True, "--honest/--no-honest",
        help="select on a discovery half, estimate on the other (post-selection guarantee)",
    ),
    lam: str = typer.Option("plugin", help="plugin|cv|<positive float>"),
    seed: int = typer.Option(0, help="RNG seed (converted once to the run's single numpy Generator)"),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """Belloni-style instrument selection (+ honest 2SLS/J/AR estimation block).

    Runs :func:`natex.iv.pipeline.discover_instruments`: plug-in Lasso
    first-stage selection over the candidate pool (reads only treatment, pool
    and controls), then — when ``--outcome`` is given — 2SLS with HC1 SEs,
    the Anderson-Rubin/Fieller confidence set and the Hansen J diagnostic on
    the estimation half. Writes ``out/instruments.json`` (NaN serialized as
    null, never 0). ``--no-honest`` selects and estimates on the full sample
    and records the post-selection caveat string in the payload.
    """
    df = pd.read_csv(csv)
    controls_list = [c.strip() for c in controls.split(",")] if controls else None
    if pool:
        pool_list = [c.strip() for c in pool.split(",")]
    else:
        excluded = {treatment, *(controls_list or []), *([outcome] if outcome else [])}
        pool_list = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]
    lam_arg: float | str = lam
    if lam not in ("plugin", "cv"):
        try:
            lam_arg = float(lam)
        except ValueError:
            typer.echo(f"--lam must be 'plugin', 'cv', or a positive float, got {lam!r}")
            raise typer.Exit(code=2) from None
    rng = np.random.default_rng(seed)
    try:
        res = discover_instruments(
            df, treatment, pool_list, outcome=outcome, controls=controls_list,
            honest=honest, lam=lam_arg, rng=rng,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    search = res.search
    loadings = np.asarray(search.loadings, dtype=float)
    est = res.estimate
    payload = _clean(
        {
            "params": {"treatment": treatment, "pool": pool_list, "controls": controls_list,
                       "outcome": outcome, "honest": honest, "lam": lam, "seed": seed,
                       "csv": str(csv)},
            "selection": {
                "selected": search.selected,
                "lam": search.lam,
                "lam_source": search.extras.get("lam_source"),
                "n_pool": len(pool_list),
                "loadings": {
                    "min": float(loadings.min()) if loadings.size else None,
                    "median": float(np.median(loadings)) if loadings.size else None,
                    "max": float(loadings.max()) if loadings.size else None,
                },
                "first_stage_F": search.first_stage_F,
                "partial_r2": search.partial_r2,
                "weak": search.weak,
                "n_iter": search.n_iter,
                "dropped_zero_variance": search.extras.get("dropped_zero_variance", []),
            },
            "split": {"honest": res.honest, "n_discovery": res.n_discovery,
                      "n_estimation": res.n_estimation},
            "estimate": None if est is None else {
                "tau": est.tau, "se": est.se, "ci": list(est.ci),
                "ar_ci": list(est.ar_ci) if est.ar_ci is not None else None,
                "ar_kind": est.ar_kind,
                "j_stat": est.j_stat, "j_p": est.j_p, "j_df": est.j_df,
                "first_stage_F": est.first_stage_F, "partial_r2": est.partial_r2,
                "weak_instrument": est.weak_instrument, "n_used": est.n_used,
            },
            # None when honest; the post-selection warning string when --no-honest.
            "caveat": res.extras.get("caveat"),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "instruments.json").write_text(json.dumps(payload, indent=1))
    names = ", ".join(search.selected) if search.selected else "(none)"
    typer.echo(f"selected {len(search.selected)}/{len(pool_list)} instruments: {names}")
    typer.echo(
        f"lam={search.lam:.3f} ({search.extras.get('lam_source', 'n/a')})  "
        f"first-stage F={search.first_stage_F:.2f}  "
        f"partial R2={search.partial_r2:.4f}  weak={search.weak}"
    )
    if res.honest:
        typer.echo(f"honest split: {res.n_discovery} discovery / {res.n_estimation} estimation rows")
    else:
        typer.echo(f"CAVEAT: {res.extras['caveat']}")
    if est is not None:
        ar = (f"AR CI=({est.ar_ci[0]:.3f},{est.ar_ci[1]:.3f})" if est.ar_ci is not None
              else f"AR set: {est.ar_kind}")
        j_txt = "n/a (just-identified)" if est.j_p is None else f"{est.j_p:.3f}"
        typer.echo(
            f"2SLS tau={est.tau:.3f} CI=({est.ci[0]:.3f},{est.ci[1]:.3f})  {ar}  J p={j_txt}"
        )
    typer.echo(f"results: {out / 'instruments.json'}")


@app.command()
def donors(
    csv: Path,
    outcome: str = typer.Option(..., help="outcome column (the panel series being matched)"),
    unit: str = typer.Option(..., help="panel unit column"),
    time: str = typer.Option(..., help="panel time column"),
    treated_unit: str = typer.Option(..., "--treated-unit", help="the treated unit's value in the UNIT column"),
    t0: float = typer.Option(..., help="first post-treatment time"),
    n_donors: int = typer.Option(
        None, "--n-donors",
        help="top-k donors by pre-trend score; default: all complete candidates",
    ),
    scoring: str = typer.Option("rmse", help="rmse|corr"),
    placebo: bool = typer.Option(
        True, "--placebo/--no-placebo", help="Abadie in-space RMSPE-ratio placebo test"
    ),
    exclude_poor_fit: float = typer.Option(
        None, "--exclude-poor-fit",
        help="drop placebos with pre-RMSPE > MULT x the treated unit's",
    ),
    out: Path = typer.Option(Path("out"), help="output directory"),
):
    """Synthetic-control donor selection, ATT and in-space placebo inference.

    Builds the unit-by-time outcome matrix, scores every complete donor
    candidate against the treated pre-trajectory, fits simplex weights on the
    top pool, and reports the counterfactual gap and post-period ATT
    (:mod:`natex.iv.donors`). With ``--placebo`` (default) every complete
    candidate is refit as pseudo-treated under the identical selection rule
    and the +1-rank two-sided RMSPE-ratio p-value is reported; ``--no-placebo``
    omits the block. Writes ``out/donors.json`` (NaN as null, never 0).
    Deterministic: no rng anywhere in the donor path.
    """
    df = pd.read_csv(csv)
    try:
        Y, units, times = unit_time_matrix(df, unit, time, outcome)
        treated: object = treated_unit
        if units.dtype.kind in "iuf" and treated_unit not in units:
            # numeric unit labels arrive as a string from the CLI
            try:
                treated = float(treated_unit)
            except ValueError:
                pass  # fall through: select_donors reports the unmatched unit
        res = select_donors(
            Y, units, times, treated, t0, n_donors=n_donors, scoring=scoring
        )
        rep = (
            sc_placebo_test(
                Y, units, times, treated, t0, n_donors=n_donors, scoring=scoring,
                exclude_poor_fit=exclude_poor_fit,
            )
            if placebo
            else None
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    placebo_block = None
    if rep is not None:
        placebo_block = {
            "p_value": rep.p_value,
            "ratio_treated": rep.ratio_treated,
            "n_used": int(rep.ratios.size),
            "n_skipped": rep.n_skipped,
            "n_placebo_candidates": rep.extras["n_placebo_candidates"],
            "poor_fit_units": rep.extras["poor_fit_units"],
            "exclude_poor_fit": exclude_poor_fit,
            # usable placebos sorted by descending post/pre RMSPE ratio
            "ratios": [
                {"unit": u, "ratio": r}
                for u, r in zip(rep.placebo_units, rep.ratios.tolist(), strict=True)
            ],
        }
    payload = _clean(
        {
            "params": {"outcome": outcome, "unit": unit, "time": time,
                       "treated_unit": treated_unit, "t0": t0, "n_donors": n_donors,
                       "scoring": scoring, "placebo": placebo,
                       "exclude_poor_fit": exclude_poor_fit, "csv": str(csv)},
            "treated_unit": res.treated_unit,
            "t0": res.t0,
            "scores": [
                {"unit": s.unit, "pre_rmse": s.pre_rmse, "pre_corr": s.pre_corr,
                 "rank": s.rank}
                for s in res.scores
            ],
            "donors": list(res.donors),
            "weights": [
                {"unit": u, "weight": w}
                for u, w in zip(res.donors, res.weights.tolist(), strict=True)
            ],
            "pre_rmspe": res.pre_rmspe,
            "post_rmspe": res.post_rmspe,
            "att_post": res.att_post,
            "times": res.times,
            "effect_by_time": res.effect_by_time,
            "diagnostics": res.extras,
            **({"placebo": placebo_block} if placebo_block is not None else {}),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "donors.json").write_text(json.dumps(payload, indent=1))
    if "failure" in res.extras:
        typer.echo(f"donor selection failed: {res.extras['failure']}")
    else:
        top = ", ".join(
            f"{u} w={w:.2f}" for u, w in zip(res.donors[:8], res.weights.tolist(), strict=False)
        )
        typer.echo(
            f"donors {len(res.donors)}/{res.extras['n_candidates']} candidates: {top}"
        )
    typer.echo(
        f"pre RMSPE={res.pre_rmspe:.3f}  post RMSPE={res.post_rmspe:.3f}  "
        f"ATT(post)={res.att_post:.3f}"
    )
    if rep is not None:
        typer.echo(
            f"placebo: p={rep.p_value:.3f}  treated ratio={rep.ratio_treated:.2f}  "
            f"usable placebos={rep.ratios.size} (skipped {rep.n_skipped})"
        )
    typer.echo(f"results: {out / 'donors.json'}")


def _discover_did(
    csv: Path,
    *,
    treatment: str,
    outcome: str | None,
    forcing: str | None,
    q: int,
    seed: int,
    degree: int,
    time: str | None,
    unit: str | None,
    bins: int,
    windows: str | None,
    restarts: int,
    method: str,
    model: str,
    out: Path,
) -> None:
    """SuDDDS branch of ``discover``: scan + validation battery + effects.

    Runs :func:`natex.did.suddds.suddds_scan`, calibrates the max-LLR with
    :func:`natex.validate.panel.panel_randomization_test` (Q=``--q``), runs
    the composition and anticipation checks on the top discovery, and — when
    an outcome column is given — :func:`natex.did.effects.did_effect` plus the
    studentized :func:`natex.did.effects.tau_randomization_test` for each of
    the dd/synthetic/gess controls. The results bundle always reports what was
    searched (windows grid, restarts, method, model, dims, bin counts —
    spec 6b obligation).
    """
    if time is None:
        typer.echo("--design did requires --time COLUMN (the panel time variable)")
        raise typer.Exit(code=2)
    ds = Dataset.from_csv(
        csv, treatment=treatment, outcome=outcome,
        forcing=forcing.split(",") if forcing else [],
        time=time, unit=unit,
    )
    window_grid = tuple(float(w) for w in windows.split(",")) if windows else None
    rng = np.random.default_rng(seed)
    panel = build_panel(ds, bins=bins)
    # Issue #10: audit-19 Bernoulli auto-matching conflicts with single_delta's
    # Gaussian profile GLR on binary treatments — resolve the default
    # combination exactly as natex.discover does (shared helper, 405a7ae);
    # an explicit --model bernoulli still raises inside suddds_scan.
    model = resolve_default_model(model, method)
    res = suddds_scan(
        ds, windows=window_grid, restarts=restarts, model=model, method=method,
        bins=bins, degree=degree, rng=rng, panel=panel,
    )
    if not res.discoveries:
        typer.echo("no qualifying discovery (no cutoff had two-sided support); nothing to report")
        raise typer.Exit(code=1)
    rand = panel_randomization_test(
        ds, res, Q=q, rng=rng, scan_kwargs={"bins": bins, "degree": degree}
    )
    top = res.discoveries[0]
    comp = composition_test(panel, top)
    # anticipation_test refits its own nuisance on the pre-period sub-panel
    # (issue #12): a full-panel background would leak the real jump into the
    # trend coefficients and fail clean discoveries.
    antic = anticipation_test(panel, top, model=res.model, degree=degree)
    effects = {}
    if ds.y is not None:
        for control in ("dd", "synthetic", "gess"):
            eff = did_effect(panel, top, control=control)
            tau_rand = tau_randomization_test(panel, top, control=control, rng=rng)
            effects[control] = {
                "tau": eff.tau, "se": eff.se, "p": tau_rand.p_value,
                "pre_mse": eff.pre_mse, "dose": eff.dose,
            }
    payload = _clean(
        {
            # Issue #29: roles recorded alongside time/unit — paper/brief read them.
            "params": {"design": "did", "treatment": treatment, "outcome": outcome,
                       "forcing": list(ds.spec.forcing),
                       "q": q, "seed": seed, "degree": degree,
                       "time": time, "unit": unit, "bins": bins, "windows": windows,
                       "restarts": restarts, "method": method, "model": model,
                       "csv": str(csv)},
            "did": {
                "scan": {"model": res.model, "method": res.method,
                         "p_value": rand.p_value,
                         "observed_max_llr": rand.observed_max_llr,
                         "null_kind": rand.null_kind},
                "discoveries": [
                    {"subset_values": d.subset_values, "t0": d.t0,
                     "window": d.window, "llr": d.llr}
                    for d in res.top(20)
                ],
                "validation": {
                    "composition_p": comp.p_value,
                    "composition_passed": comp.passed,
                    "anticipation_shifts": list(antic.shifts),
                    "anticipation_p_holm": antic.p_holm.tolist(),
                    "anticipation_passed": antic.passed,
                },
                "effects": effects,
                # spec 6b: always report what was searched.
                "searched": {
                    "windows": list(res.windows), "restarts": res.restarts,
                    "method": res.method, "model": res.model,
                    "dims": list(panel.dim_names),
                    "bin_counts": dict(zip(panel.dim_names, panel.dim_sizes)),
                },
            },
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    typer.echo(f"model={res.model}  max LLR={rand.observed_max_llr:.2f}  scan p={rand.p_value:.3f}")
    typer.echo(f"top subset: {top.subset_values}  t0={top.t0:g}  W={top.window:g}")
    typer.echo(f"composition passed: {comp.passed}   anticipation passed: {antic.passed}")
    if effects:
        e = effects["dd"]
        typer.echo(
            f"dd tau={e['tau']:.3f} p={e['p']:.3f} "
            f"(synthetic tau={effects['synthetic']['tau']:.3f}, "
            f"gess tau={effects['gess']['tau']:.3f})"
        )
    typer.echo(f"results: {out / 'results.json'}")


@app.command()
def paper(
    bundle: Path = typer.Option(..., "--bundle",
        help="results bundle dir (ResultsBundle.save, or a discover --out dir)"),
    format: str = typer.Option("md", help="md|latex; latex also compiles when tectonic is on PATH"),
    out: Path = typer.Option(None, help="output dir; default BUNDLE/paper"),
):
    """Render the AI-draft paper from a results bundle (markdown always works)."""
    if format not in ("md", "latex"):
        typer.echo(f"--format must be md or latex, got {format!r}")
        raise typer.Exit(code=2)
    try:
        loaded = ResultsBundle.load(bundle)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # ValueError covers json.JSONDecodeError (its subclass).
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    try:
        result = render_paper(loaded, format=format, out_dir=out)
    except ImportError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    artifact = result.markdown if result.markdown is not None else result.tex
    typer.echo(f"paper: {artifact}")
    if result.pdf is not None:
        typer.echo(f"pdf: {result.pdf}")
    elif result.tex is not None:
        typer.echo(result.message)  # tectonic skip/failure message, verbatim
    typer.echo("review before sharing — AI-generated draft")


@app.command()
def brief(
    bundle: Path = typer.Option(..., "--bundle",
        help="results bundle dir (ResultsBundle.save, or a discover --out dir)"),
    out: Path = typer.Option(None,
        help="output dir or .md path; default BUNDLE/research-brief.md"),
):
    """Write the deep-research handoff brief (research-brief.md) from a results bundle."""
    try:
        loaded = ResultsBundle.load(bundle)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # ValueError covers json.JSONDecodeError (its subclass).
        typer.echo(str(exc))
        raise typer.Exit(code=2) from None
    path = research_brief(loaded, out if out is not None else bundle)
    typer.echo(f"brief: {path}")
    typer.echo("hand this file to your deep-research tooling; verify everything it returns")
