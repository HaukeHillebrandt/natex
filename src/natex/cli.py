"""natex command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from natex.data.registry import REGISTRY, verify
from natex.data.spec import Dataset
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan
from natex.scan.coarse import coarse_to_fine_scan
from natex.validate.density import density_test
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
    out: Path = typer.Option(Path("out")),
):
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
