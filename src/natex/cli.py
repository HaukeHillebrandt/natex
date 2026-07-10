"""natex command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from natex.data.spec import Dataset
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan
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
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


@app.command()
def discover(
    csv: Path,
    treatment: str = typer.Option(...),
    outcome: str = typer.Option(None),
    forcing: str = typer.Option(None, help="comma-separated; default: all numeric"),
    k: int = typer.Option(50),
    q: int = typer.Option(99, help="randomization replicas"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("out")),
):
    ds = Dataset.from_csv(
        csv, treatment=treatment, outcome=outcome,
        forcing=forcing.split(",") if forcing else None,
    )
    rng = np.random.default_rng(seed)
    res = lord3_scan(ds, k=k, rng=rng)
    rand = randomization_test(ds, res, Q=q, rng=rng, scan_kwargs={"k": k})
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
            "params": {"k": k, "q": q, "seed": seed, "csv": str(csv)},
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
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    typer.echo(f"model={res.model}  max LLR={rand.observed_max_llr:.2f}  scan p={rand.p_value:.3f}")
    typer.echo(f"top center (raw z): {ds.Z[top.center_index]}")
    typer.echo(f"placebo passed: {placebo.passed}   density p: {dens.p_value:.3f}")
    if effects:
        e = effects["2sls"]
        typer.echo(f"2SLS tau={e['tau']:.3f} CI=({e['ci'][0]:.3f},{e['ci'][1]:.3f}) weak_iv={e['weak_instrument']}")
    typer.echo(f"results: {out / 'results.json'}")
