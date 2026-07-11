"""Synthetic benchmark harness reproducing the KDD-2018 ch.5 evaluation curves.

``nig_power_curve`` sweeps the planted discontinuity strength zeta and reports,
per (zeta, degree, model) cell: NIG of the top-scoring neighborhood against the
known region D, power at level alpha under the fitted-null Monte Carlo test
(+1-rank p-values — a parametric bootstrap, never "exact"; audit item 1), and
NaN-safe medians of the 2SLS/Wald effect estimates. ``label_noise_curve``
implements the KDD fuzzification protocol P(T_rho = T) = rho on a sharp-ish
binary synthetic.

Reproducibility contract: every experiment draws from child generators spawned
from ``default_rng(seed)`` keyed by fixed (sweep-point, experiment) indices, so
each row of the returned frame is independent of evaluation order and the whole
frame is bit-identical across runs with the same arguments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from natex.data.spec import Dataset
from natex.data.synthetic import inject_label_noise, make_synthetic
from natex.estimate.local2sls import local_2sls, wald_estimate
from natex.rdd.lord3 import lord3_scan
from natex.rdd.metrics import normalized_information_gain
from natex.scan.geometry import build_geometry
from natex.validate.randomization import randomization_test

_NAN = float("nan")

CURVE_COLUMNS = [
    "zeta",
    "degree",
    "model",
    "kind",
    "nig_mean",
    "nig_se",
    "power",
    "p_mean",
    "tau_2sls_median",
    "tau_wald_median",
    "n_experiments",
]


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else _NAN


def _nanse(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return _NAN
    return float(finite.std(ddof=1) / math.sqrt(finite.size))


def _nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).any():
        return float(np.nanmedian(arr))
    return _NAN


def _resolve_model(model: str, kind: str) -> str:
    if model == "auto":
        return "bernoulli" if kind == "binary" else "normal"
    return model


def _run_cell(
    ds: Dataset,
    D: np.ndarray,
    geometry,
    k: int,
    model: str,
    degree: int,
    Q: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One (dataset, degree, model) evaluation: top-1 NIG, +1-rank p, tau-hats."""
    res = lord3_scan(ds, k=k, model=model, degree=degree, geometry=geometry)
    if not res.discoveries:
        # Degenerate scan (e.g. globally homogeneous binary T): NaN, never 0.0.
        return {"nig": _NAN, "p": _NAN, "tau_2sls": _NAN, "tau_wald": _NAN}
    top = res.discoveries[0]
    nig = normalized_information_gain(D, top.members, top.group1)
    rep = randomization_test(
        ds, res, Q=Q, rng=rng, scan_kwargs={"k": k, "degree": degree}, geometry=geometry
    )
    return {
        "nig": float(nig),
        "p": float(rep.p_value),
        "tau_2sls": local_2sls(ds, top).tau,
        "tau_wald": wald_estimate(ds, top).tau,
    }


def nig_power_curve(
    kind: str,
    zetas: Sequence[float],
    n_experiments: int = 50,
    n: int = 1000,
    k: int = 50,
    degrees: Sequence[int] = (1, 2, 4),
    models: Sequence[str] = ("auto",),
    Q: int = 99,
    alpha: float = 0.05,
    tau: float = 5.0,
    boundary: float | str = "random",
    heteroskedastic: bool = True,
    confounder: str = "uniform",
    seed: int = 0,
) -> pd.DataFrame:
    """NIG / power / tau-hat vs zeta, one row per (zeta, degree, model) cell.

    Per experiment one synthetic dataset is drawn and shared by every
    (degree, model) cell, matching the paper's protocol and making the
    Bernoulli-vs-Normal comparison (Fig 7) a paired one. ``power`` is the
    fraction of experiments whose +1-rank Monte Carlo p-value is <= alpha
    (a failed cell — no discoveries — counts as a non-rejection).
    """
    if kind not in ("real", "binary"):
        raise ValueError(f"kind must be 'real' or 'binary', got {kind!r}")
    zetas = list(zetas)
    cells = [(int(d), _resolve_model(m, kind)) for d in degrees for m in models]
    children = np.random.default_rng(seed).spawn(len(zetas) * n_experiments)

    rows: list[dict] = []
    for iz, zeta in enumerate(zetas):
        results: dict[tuple[int, str], list[dict[str, float]]] = {c: [] for c in cells}
        for ie in range(n_experiments):
            streams = children[iz * n_experiments + ie].spawn(1 + len(cells))
            ds, D = make_synthetic(
                n=n,
                zeta=float(zeta),
                tau=tau,
                kind=kind,
                rng=streams[0],
                boundary=boundary,
                heteroskedastic=heteroskedastic,
                confounder=confounder,
            )
            geometry = build_geometry(ds.Z_std, k)
            for ic, (degree, model) in enumerate(cells):
                results[(degree, model)].append(
                    _run_cell(ds, D, geometry, k, model, degree, Q, streams[1 + ic])
                )
        for degree, model in cells:
            cell = results[(degree, model)]
            ps = np.asarray([c["p"] for c in cell], dtype=float)
            rejected = np.where(np.isfinite(ps), ps, np.inf) <= alpha
            rows.append(
                {
                    "zeta": float(zeta),
                    "degree": degree,
                    "model": model,
                    "kind": kind,
                    "nig_mean": _nanmean([c["nig"] for c in cell]),
                    "nig_se": _nanse([c["nig"] for c in cell]),
                    "power": float(rejected.mean()),
                    "p_mean": _nanmean(ps),
                    "tau_2sls_median": _nanmedian([c["tau_2sls"] for c in cell]),
                    "tau_wald_median": _nanmedian([c["tau_wald"] for c in cell]),
                    "n_experiments": n_experiments,
                }
            )
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def label_noise_curve(
    rhos: Sequence[float],
    n_experiments: int = 25,
    n: int = 2000,
    k: int = 50,
    zeta_sharp: float = 8.0,
    seed: int = 0,
) -> pd.DataFrame:
    """KDD noise-injection protocol on a sharp-ish binary synthetic.

    ``zeta_sharp`` makes T essentially equal to the region indicator D; each
    experiment then replaces T with T_rho such that P(T_rho = T) = rho and
    reports the top-1 NIG of a Bernoulli scan against D. rho = 1 recovers the
    sharp design; rho = 0.5 destroys all signal.
    """
    rhos = list(rhos)
    children = np.random.default_rng(seed).spawn(len(rhos) * n_experiments)
    rows: list[dict] = []
    for ir, rho in enumerate(rhos):
        nigs: list[float] = []
        for ie in range(n_experiments):
            data_rng, noise_rng = children[ir * n_experiments + ie].spawn(2)
            ds, D = make_synthetic(
                n=n,
                zeta=float(zeta_sharp),
                kind="binary",
                rng=data_rng,
                boundary="random",
                heteroskedastic=True,
                confounder="uniform",
            )
            t_rho = inject_label_noise(ds.T, float(rho), noise_rng)
            df_noisy = ds.df.copy()
            df_noisy[ds.spec.treatment] = t_rho.astype(float)
            ds_noisy = Dataset(df_noisy, ds.spec)
            res = lord3_scan(ds_noisy, k=k, model="bernoulli")
            if res.discoveries:
                top = res.discoveries[0]
                nigs.append(float(normalized_information_gain(D, top.members, top.group1)))
            else:
                nigs.append(_NAN)
        rows.append(
            {
                "rho": float(rho),
                "nig_mean": _nanmean(nigs),
                "nig_se": _nanse(nigs),
                "n_experiments": n_experiments,
            }
        )
    return pd.DataFrame(rows, columns=["rho", "nig_mean", "nig_se", "n_experiments"])
