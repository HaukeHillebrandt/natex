"""natex command-line interface."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import typer

from natex.data.registry import REGISTRY, data_root, verify
from natex.data.spec import Dataset
from natex.dee.debias import dee_debias
from natex.dee.vknn import select_m_prime
from natex.did.background import fit_did_background
from natex.did.effects import did_effect, tau_randomization_test
from natex.did.panel import build_panel
from natex.did.suddds import suddds_scan
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan
from natex.scan.coarse import coarse_to_fine_scan
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


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


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


@app.command()
def discover(
    csv: Path,
    treatment: str = typer.Option(...),
    outcome: str = typer.Option(None),
    forcing: str = typer.Option(None, help="comma-separated; default: all numeric"),
    k: int = typer.Option(50),
    q: int = typer.Option(99, help="randomization replicas"),
    seed: int = typer.Option(0),
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
        "auto", help="auto|normal|bernoulli (--design did; audit-19 model matching)"
    ),
    out: Path = typer.Option(Path("out")),
):
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
    coarse_block = None
    if coarse:
        ctf = coarse_to_fine_scan(ds, k=k, n_coarse=n_coarse, degree=degree, rng=rng)
        # Validation/estimation below operate on the fine-stage (full-resolution)
        # result; the coverage block reports what was and wasn't searched (spec 6b).
        res = ctf.result
        coarse_block = {"frac_centers_scanned": ctf.frac_centers_scanned, **ctf.params}
    else:
        res = lord3_scan(ds, k=k, degree=degree, rng=rng)
    rand = randomization_test(ds, res, Q=q, rng=rng, scan_kwargs={"k": k, "degree": degree})
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
            "params": {"k": k, "q": q, "seed": seed, "degree": degree,
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
def debias(
    csv: Path,
    treatment: str = typer.Option(...),
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
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("out")),
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
    background = fit_did_background(panel, model=res.model, degree=degree)
    antic = anticipation_test(panel, background, top)
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
            "params": {"design": "did", "q": q, "seed": seed, "degree": degree,
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
