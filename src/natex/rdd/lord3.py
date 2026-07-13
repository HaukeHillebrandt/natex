"""LoRD3: local regression-discontinuity discovery (Herlands et al. 2018),
reimplemented per docs/math_audit_final.md."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeWarning
from scipy.special import logit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from natex.data.spec import Dataset
from natex.scan.geometry import ScanGeometry, build_geometry
from natex.scan.neighborhoods import local_residual_variance
from natex.scan.statistics import bernoulli_llr_all_splits, normal_llr_all_splits

_P_CLIP = 1e-6


@dataclass
class Discovery:
    center_index: int
    k: int
    llr: float
    normal: np.ndarray
    members: np.ndarray
    group1: np.ndarray
    p_value: float | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class LoRD3Result:
    discoveries: list[Discovery]
    model: str
    k: int
    centers: np.ndarray | None = None

    def top(self, m: int) -> list[Discovery]:
        return self.discoveries[:m]


def fit_treatment_model(X: np.ndarray, T: np.ndarray, model: str, degree: int):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)
    if model == "normal":
        est = LinearRegression().fit(Xp, T)
        return lambda A: est.predict(poly.transform(A)), "normal"
    # Ridge-penalized logistic on standardized features. Scale-invariant guard
    # against perfect separation (sharp designs): the unpenalized MLE does not
    # exist there, the fitted background absorbs the discontinuity, and the
    # Bernoulli(p-hat) null replicas degenerate into exact copies of T. Penalized
    # likelihood under separation follows the audit's remedy (Firth-style); the
    # standardization makes the penalty independent of covariate units.
    scaler = StandardScaler().fit(Xp)
    with warnings.catch_warnings():
        # sklearn <= 1.6 still passes the removed 'iprint' option to scipy's
        # L-BFGS-B — a known, data-independent upstream bug that fires once
        # PER FIT (once per null replica). scipy >= 1.18 reports it as an
        # OptimizeWarning ("Unknown solver options"); scipy 1.17 (the locked
        # Python 3.11 resolution) as a DeprecationWarning. Suppress exactly
        # those messages; convergence/separation warnings stay visible
        # (issue #5).
        warnings.filterwarnings(
            "ignore", message="Unknown solver options: iprint", category=OptimizeWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r"scipy\.optimize: The `disp` and `iprint` options",
            category=DeprecationWarning,
        )
        est = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(Xp), T.astype(int))
    return (
        lambda A: est.predict_proba(scaler.transform(poly.transform(A)))[:, 1],
        "bernoulli",
    )


def lord3_scan(
    dataset: Dataset,
    k: int = 50,
    model: str = "auto",
    degree: int = 1,
    rng: np.random.Generator | None = None,
    geometry: ScanGeometry | None = None,
    centers: np.ndarray | None = None,
) -> LoRD3Result:
    if model == "auto":
        model = "bernoulli" if dataset.treatment_is_binary else "normal"
    if model == "bernoulli":
        classes = np.unique(dataset.T[~np.isnan(dataset.T)])
        if classes.size < 2:
            # Diagnose instead of surfacing sklearn's raw "needs samples of at
            # least 2 classes" error (dogfood finding): Dataset listwise-deletes
            # rows with NaN in any forcing/covariate column, which can silently
            # remove every row on one side of a cutoff.
            top_loss = dataset.top_row_loss()  # issue #1: name the offenders
            detail = ""
            if top_loss:
                detail = "; top row-loss columns: " + ", ".join(
                    f"{c}: {lost}/{dataset.n_rows_input} rows"
                    for c, lost in top_loss.items()
                )
            raise ValueError(
                f"treatment '{dataset.spec.treatment}' has a single class "
                f"({classes.tolist()}) across the {dataset.n} scan rows -- "
                "Dataset drops rows with NaN/inf in any forcing/covariate "
                "column, which can delete every row on one side of a cutoff; "
                "check covariate missingness (or drop the offending "
                f"covariates){detail}"
            )
    X, T, Z = dataset.X, dataset.T, dataset.Z_std
    predict, kind = fit_treatment_model(X, T, model, degree)
    if geometry is None:
        geometry = build_geometry(Z, k)
    elif geometry.k != k:
        raise ValueError(f"geometry.k={geometry.k} disagrees with k={k}")
    idx = geometry.idx

    if kind == "normal":
        r = T - predict(X)
        # Full-geometry variances even when scanning a center subset: every
        # member's OWN-kNN variance is needed, not just the centers'.
        sigma2 = local_residual_variance(r, idx)
    else:
        p_hat = np.clip(predict(X), _P_CLIP, 1.0 - _P_CLIP)
        eta = logit(p_hat)

    discoveries: list[Discovery] = []
    n = Z.shape[0]
    center_iter = range(n) if centers is None else np.asarray(centers, dtype=int)
    for i in center_iter:
        i = int(i)
        members = idx[i]
        if kind == "bernoulli":
            tm = T[members]
            if tm.min() == tm.max():
                # Homogeneous fast path: with constant T every split side is
                # pure, so each split's ll1 and the null's boundary supremum
                # are both 0 and every LLR is provably 0.0 (audit item 21).
                # Skip the kernel; recording nothing for this center matches
                # the slow path's all-zero scores. Null replicas hit the same
                # branch, keeping observed/null scans consistent.
                continue
        cz = Z[members] - Z[i]
        G, keep = geometry.partitions_for(i, Z)
        if G.shape[1] == 0:
            continue
        if kind == "normal":
            llrs = normal_llr_all_splits(r[members], 1.0 / sigma2[members], G)
        else:
            llrs = bernoulli_llr_all_splits(T[members], eta[members], G)
        j = int(np.argmax(llrs))
        raw_normal = cz[keep[j]]
        norm = float(np.linalg.norm(raw_normal))
        discoveries.append(
            Discovery(
                center_index=i,
                k=k,
                llr=float(llrs[j]),
                normal=raw_normal / norm if norm > 0 else raw_normal,
                members=members.copy(),
                group1=G[:, j].copy(),
            )
        )
    discoveries.sort(key=lambda d: d.llr, reverse=True)
    return LoRD3Result(discoveries=discoveries, model=kind, k=k, centers=centers)
