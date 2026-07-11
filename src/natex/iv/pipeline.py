"""Honest instrument-discovery pipeline (audit item 1 lineage).

``discover_instruments`` splits the rows with ``validate.honest.honest_split``:
``select_instruments`` sees ONLY the discovery half; ``iv_2sls`` (2SLS + Hansen
J + Anderson-Rubin set, on the selected columns) sees ONLY the estimation
half. Because the two halves are disjoint, the selection event is a function
of the discovery half alone and is therefore independent of the estimation
noise — so the J/AR/Wald p-values computed on the estimation half retain
their nominal distributions with NO post-selection correction. This sample
splitting is the pipeline's post-selection guarantee.

``honest=False`` selects and estimates on the FULL sample and sets
``extras["caveat"]``: post-selection inference is then NOT corrected — the
same noise that picked the instruments enters the p-values, which are
optimistic to an uncontrolled degree.

Exclusion remains untestable either way (audit item 3 lineage): the
estimation-half Hansen J tests only the overidentifying restrictions given at
least one valid instrument; ``j_p`` is None when just-identified, never a
fabricated value. Audit item 10: first-stage strength is re-checked on the
estimation half by ``iv_2sls`` (HC1 F, ``weak_instrument``) — discovery-half
selection never implies estimation-half relevance.

NaN policy (spec section 5 item 8): an empty selection yields a NaN
``IVEstimate`` with ``extras["reason"] = "empty selection"`` — never 0.0.
Row-level NaN handling is delegated to the components, which drop non-finite
rows and count them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from natex.estimate.iv2sls import IVEstimate, iv_2sls
from natex.iv.search import InstrumentSearchResult, select_instruments
from natex.validate.honest import honest_split


@dataclass
class InstrumentDiscovery:
    search: InstrumentSearchResult  # fitted on the discovery half (or full sample)
    estimate: IVEstimate | None  # 2SLS + J + AR on the estimation half; None if no outcome
    honest: bool
    n_discovery: int
    n_estimation: int
    extras: dict = field(default_factory=dict)


_DISHONEST_CAVEAT = (
    "post-selection inference not corrected: instruments were selected and "
    "estimated on the same sample, so J/AR/Wald p-values are optimistic"
)


def discover_instruments(
    df: pd.DataFrame,
    treatment: str,
    pool: list[str],
    outcome: str | None = None,
    controls: list[str] | None = None,
    honest: bool = True,
    frac_discovery: float = 0.5,
    lam: float | str = "plugin",
    rng: np.random.Generator | None = None,
) -> InstrumentDiscovery:
    """Select instruments on a discovery half, estimate + J + AR on the rest.

    ``rng`` is required when ``honest=True`` (it draws the row split) or when
    ``lam="cv"`` (fold shuffling); ``honest=False`` with the RNG-free plug-in
    lambda needs none. Split indices (positional row indices into ``df``) are
    reported in ``extras["idx_discovery"]`` / ``extras["idx_estimation"]``.
    ``outcome=None`` runs selection only (discovery never reads the outcome)
    and returns ``estimate=None``. See the module docstring for the honesty
    guarantee and its ``honest=False`` caveat.
    """
    missing = [
        col
        for col in [treatment, *pool, *(controls or []), *([outcome] if outcome else [])]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"column(s) not in df: {missing}")
    if not 0.0 < frac_discovery < 1.0:
        raise ValueError(f"frac_discovery must be in (0, 1), got {frac_discovery}")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    if honest and rng is None:
        raise ValueError("pass an explicit numpy Generator (honest=True draws the row split)")

    n = len(df)
    t_all = df[treatment].to_numpy(dtype=float)
    pool_all = df[list(pool)].to_numpy(dtype=float)
    c_full = df[list(controls)].to_numpy(dtype=float) if controls else None

    extras: dict = {}
    if honest:
        idx_d, idx_e = honest_split(n, frac_discovery, rng=rng)
    else:
        idx_d = idx_e = np.arange(n)
        extras["caveat"] = _DISHONEST_CAVEAT
    extras["idx_discovery"] = idx_d
    extras["idx_estimation"] = idx_e

    search = select_instruments(
        t_all[idx_d],
        pool_all[idx_d],
        controls=c_full[idx_d] if c_full is not None else None,
        pool_names=list(pool),
        lam=lam,
        rng=rng,
    )

    estimate: IVEstimate | None = None
    if outcome is not None:
        y_e = df[outcome].to_numpy(dtype=float)[idx_e]
        z_cols = [pool.index(name) for name in search.selected]
        estimate = iv_2sls(
            y_e,
            t_all[idx_e],
            pool_all[idx_e][:, z_cols],
            controls=c_full[idx_e] if c_full is not None else None,
        )
        if not search.selected:
            estimate.extras["reason"] = "empty selection"

    return InstrumentDiscovery(
        search=search,
        estimate=estimate,
        honest=honest,
        n_discovery=int(len(idx_d)),
        n_estimation=int(len(idx_e)),
        extras=extras,
    )
