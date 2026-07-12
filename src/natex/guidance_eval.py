"""Blind-vs-informed guidance evaluation scaffold (phase llm-analyst task 10).

Gate question: does an informed guidance backend put the TRUE design
configuration earlier in the search plan (lower ``rank_of_truth``) than the
blind :class:`~natex.llm.NullBackend` heuristics? Each :class:`EvalCase`
buries a known synthetic truth behind decoys — a binary decoy column inserted
BEFORE the true treatment (so Null's column-order heuristics rank the decoy
treatment first) plus one pure-noise numeric column — and carries a free-text
``context`` hint only an informed backend can exploit.

Importable logic only, NO file IO (mirrors the ``natex.benchmarks`` /
``benchmarks/run_*.py`` split); the manual runner with API arms lives in
``benchmarks/guidance_eval.py`` and never runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from natex.data.synthetic import make_synthetic
from natex.data.synthetic_did import make_did_synthetic
from natex.intake.analyst import study
from natex.intake.plans import SearchPlan
from natex.llm import GuidanceBackend

# Fixed decoy names: binary flags a blind heuristic cannot tell from a real
# treatment. None may match the profiler's time regex (date|time|year|month|
# quarter) or NullBackend's count-column regex (n|count|pop|...).
DECOY_NAMES = ("holiday", "eligible_flag", "promo_flag", "audit_flag", "pilot_flag", "waiver")

EVAL_COLUMNS = [
    "case",
    "design",
    "n_candidates_null",
    "n_candidates_backend",
    "rank_null",
    "rank_backend",
]

_DID_YEAR0 = 2000  # make_did_synthetic times 1..periods -> calendar years


@dataclass
class EvalCase:
    """One dataset with a known true design buried among decoys."""

    name: str
    df: pd.DataFrame  # true design buried among decoys
    context: str  # free-text hint an informed backend can exploit
    design: str  # "rdd" | "did"
    treatment: str
    forcing: tuple[str, ...]  # rdd truth ((,) empty for did)
    time: str | None  # did truth (None for rdd)


def _rdd_case(i: int, rng: np.random.Generator) -> EvalCase:
    ds, _ = make_synthetic(n=400, px=3, pz=2, zeta=6.0, kind="binary", rng=rng)
    decoy = DECOY_NAMES[i % len(DECOY_NAMES)]
    if i >= len(DECOY_NAMES):
        decoy = f"{decoy}_{i}"
    # y is placed FIRST so NullBackend's first-outcome-guess heuristic takes y
    # as the outcome and leaves x0 available as forcing: the blind plan then
    # CONTAINS the truth at a strictly worse rank (an integer >= 1) instead of
    # missing it entirely, which is what the eval measures.
    df = ds.df[["y", "x0", "x1", "x2", "T"]].copy()
    df.insert(df.columns.get_loc("T"), decoy, rng.integers(0, 2, len(df)))
    df["noise"] = rng.normal(0.0, 1.0, size=len(df))  # pure-noise numeric decoy
    context = (
        "Individuals are enrolled in program T when their scores x0 and x1 both "
        f"clear an eligibility threshold; '{decoy}' is an unrelated administrative "
        "flag and 'noise' is a meaningless index. The outcome of interest is y."
    )
    return EvalCase(
        name=f"rdd-{i}-{decoy}", df=df, context=context,
        design="rdd", treatment="T", forcing=("x0", "x1"), time=None,
    )


def _did_case(rng: np.random.Generator) -> EvalCase:
    ds, _ = make_did_synthetic(n=400, d=3, V=4, periods=6, theta_kind="binary", rng=rng)
    df = ds.df.copy()
    year = _DID_YEAR0 + df.pop("t").astype(int)  # time-like by name AND value range
    df.insert(0, "year", year)
    # String unit ids, unique within each year: the profiler then sees the
    # (unit, year) grid as a panel candidate, so Null proposes a did design.
    df.insert(0, "unit", "u" + df.groupby("year").cumcount().astype(str))
    context = (
        "Repeated cross-section of units observed over calendar years; the binary "
        "program theta switched on for some covariate profiles after a known year. "
        "The outcome of interest is y."
    )
    return EvalCase(
        name="did-0", df=df, context=context,
        design="did", treatment="theta", forcing=(), time="year",
    )


def make_eval_cases(
    n_rdd: int = 4,
    include_did: bool = True,
    rng: np.random.Generator | None = None,
) -> list[EvalCase]:
    """Build the eval suite: ``n_rdd`` decoy-laden rdd cases (+ one did case).

    rdd truth is ``("rdd", "T", ("x0", "x1"))`` with a fixed-name binary decoy
    inserted before ``T`` and a pure-noise numeric column appended; did truth
    is matched on ``(design, treatment, time)`` only.
    """
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (reproducibility contract)")
    cases = [_rdd_case(i, rng) for i in range(n_rdd)]
    if include_did:
        cases.append(_did_case(rng))
    return cases


def rank_of_truth(plan: SearchPlan, case: EvalCase) -> int | None:
    """Index in ``plan.ranked()`` of the first candidate matching the truth.

    rdd: design + treatment match and candidate forcing is a superset-or-equal
    of the true forcing; did: design + treatment + time match. ``None`` if the
    truth is absent from the plan (worse than any finite rank).
    """
    for i, c in enumerate(plan.ranked()):
        if c.design != case.design or c.treatment != case.treatment:
            continue
        if case.design == "rdd":
            if set(case.forcing) <= set(c.forcing):
                return i
        elif c.time == case.time:
            return i
    return None


def run_guidance_eval(
    make_backend: Callable[[EvalCase], GuidanceBackend | None],
    n_rdd: int = 4,
    include_did: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Run both arms on every case; one row per case, columns ``EVAL_COLUMNS``.

    Blind arm: ``study(df, context, guidance=None, rng=default_rng(seed))``
    (NullBackend heuristics). Informed arm: same call with
    ``guidance=make_backend(case)`` — the factory builds a FRESH backend per
    case (``None`` means the blind arm twice, the runner's smoke mode). Ranks
    are nullable ``Int64`` (absent truth -> ``pd.NA``). No file IO here.
    """
    cases = make_eval_cases(n_rdd=n_rdd, include_did=include_did, rng=np.random.default_rng(seed))
    rows = []
    for case in cases:
        blind = study(case.df, context=case.context, guidance=None, rng=np.random.default_rng(seed))
        informed = study(
            case.df, context=case.context,
            guidance=make_backend(case), rng=np.random.default_rng(seed),
        )
        rows.append(
            {
                "case": case.name,
                "design": case.design,
                "n_candidates_null": len(blind.search_plan.candidates),
                "n_candidates_backend": len(informed.search_plan.candidates),
                "rank_null": rank_of_truth(blind.search_plan, case),
                "rank_backend": rank_of_truth(informed.search_plan, case),
            }
        )
    frame = pd.DataFrame(rows, columns=EVAL_COLUMNS)
    for col in ("rank_null", "rank_backend"):
        frame[col] = frame[col].astype("Int64")
    return frame
