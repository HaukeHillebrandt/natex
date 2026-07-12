#!/usr/bin/env python
"""Blind-vs-informed guidance eval runner (phase llm-analyst task 10; manual).

Gate question: does the informed plan hit the true design configuration
EARLIER (lower rank of truth in the search plan) than the blind NullBackend
plan? Every case runs both arms via ``natex.guidance_eval.run_guidance_eval``;
one CSV row per case with ``rank_null`` vs ``rank_backend`` (empty cell =
truth absent from that arm's plan, worse than any finite rank).

Arms (``--backend``):

- ``null``    — ``make_backend = lambda case: None``: the blind arm twice, a
  smoke mode for the scaffold itself (ranks must coincide).
- ``agent``   — file-based AgentBackend under ``--workdir`` (one subdirectory
  per case); a calling coding agent answers the request files.
- ``anthropic`` / ``gemini`` — API backends (``uv sync --extra llm`` plus the
  provider key in the environment). API arms are MANUAL ONLY and never run in
  CI; the CI slice (``tests/test_guidance_eval.py``) uses MockBackend only.

Run e.g.::

    uv run python benchmarks/guidance_eval.py --backend null
    uv run python benchmarks/guidance_eval.py --backend anthropic --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from natex.guidance_eval import EvalCase, run_guidance_eval
from natex.llm import GuidanceBackend

_HERE = Path(__file__).resolve().parent


def make_backend_factory(
    backend: str,
    model: str | None,
    workdir: str | Path,
) -> Callable[[EvalCase], GuidanceBackend | None]:
    """One FRESH backend per case (file sequences / canned state never shared)."""
    if backend == "null":
        return lambda case: None  # blind arm twice: smoke mode
    if backend == "agent":
        from natex.llm import AgentBackend

        root = Path(workdir)
        return lambda case: AgentBackend(root / case.name)
    if backend == "anthropic":
        from natex.llm import AnthropicBackend

        return lambda case: AnthropicBackend(**({"model": model} if model else {}))
    if backend == "gemini":
        from natex.llm import GeminiBackend

        return lambda case: GeminiBackend(**({"model": model} if model else {}))
    raise ValueError(f"unknown backend {backend!r}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend", choices=("null", "agent", "anthropic", "gemini"), default="null"
    )
    parser.add_argument("--model", default=None, help="API model id (backend default if omitted)")
    parser.add_argument(
        "--workdir",
        default=str(_HERE / "out" / "guidance_agent"),
        help="AgentBackend request/response root (one subdirectory per case)",
    )
    parser.add_argument("--n-rdd", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(_HERE / "out" / "guidance_eval.csv"))
    args = parser.parse_args(argv)

    frame = run_guidance_eval(
        make_backend_factory(args.backend, args.model, args.workdir),
        n_rdd=args.n_rdd,
        seed=args.seed,
    )
    print(frame.to_string(index=False))
    for arm in ("null", "backend"):
        ranks = frame[f"rank_{arm}"]
        mean = ranks.mean()  # nullable Int64: NA (truth absent) skipped
        print(
            f"mean rank_{arm} = {float(mean):.3f} "
            f"(truth found in {int(ranks.notna().sum())}/{len(frame)} plans)"
            if ranks.notna().any()
            else f"mean rank_{arm} = n/a (truth absent from every plan)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
