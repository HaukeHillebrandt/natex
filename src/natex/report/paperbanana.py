"""Optional paperbanana adapter: one method-pipeline diagram per bundle.

:func:`_pipeline_description` is PURE deterministic text built from
``bundle.results`` only — the natex pipeline as run (LoRD3/SuDDDS by name,
coverage counts, validation battery, estimators, figure list) — and is fully
unit-testable on a core-only install. :func:`generate_method_diagram` lazily
imports ``paperbanana`` (the ``paperbanana`` extra; the library needs its own
provider key) and calls its documented SINGLE entry point::

    paperbanana.generate_diagram(description=<str>, output_path=str(out))

This adapter is the ONLY place to update if the real library's API differs
from that contract — CI tests it against a monkeypatched fake module, never a
real API call.
"""

from __future__ import annotations

from pathlib import Path

from natex.report.bundle import ResultsBundle

_PAPERBANANA_EXTRA_MSG = (
    "method diagrams require the paperbanana extra: "
    'pip install "natex-discovery[paperbanana]"'
)

# design -> discovery method name (spec section 7: name the methods, not codes)
_METHOD_NAMES = {
    "rdd": "LoRD3 local regression-discontinuity scan",
    "did": "SuDDDS subset difference-in-differences scan",
}

# fixed order keeps the description deterministic across bundles
_ESTIMATOR_LABELS = (
    ("2sls", "2SLS"),
    ("wald", "Wald"),
    ("dd", "difference-in-differences"),
    ("synthetic", "synthetic control"),
    ("gess", "GESS"),
)

_VALIDATION_STAGES = {
    "rdd": (
        "Monte Carlo randomization inference on the scan statistic",
        "Holm-adjusted covariate placebo tests",
        "McCrary-style density (manipulation) test",
    ),
    "did": (
        "Monte Carlo randomization inference on the scan statistic",
        "composition-stability check",
        "anticipation (pre-trend) check",
    ),
}


def _pipeline_description(bundle: ResultsBundle) -> str:
    """Pure text: the natex pipeline as run on this bundle.

    Deterministic (same bundle -> byte-identical string); reads ONLY
    ``bundle.results``; no paperbanana import, no rng, no I/O.
    """
    r = bundle.results
    configs = [c for c in (r.get("configs") or []) if isinstance(c, dict)]
    scanned = [c for c in configs if c.get("status") == "scanned"]
    pool = scanned or configs

    designs: list[str] = []
    for cfg in pool:
        design = (cfg.get("candidate") or {}).get("design")
        if design and design not in designs:
            designs.append(design)

    methods = [_METHOD_NAMES.get(d, d) for d in designs]
    lines = [
        "natex automated natural-experiment discovery pipeline, as run:",
        "1. Discovery scan (never reads the outcome): "
        + ("; ".join(methods) if methods else "no designs recorded")
        + ".",
    ]

    searched = r.get("searched")
    if isinstance(searched, dict):
        n = {
            key: int(searched.get(key) or 0)
            for key in (
                "n_total", "n_scanned", "n_skipped_budget", "n_failed", "n_invalid",
            )
        }
        lines.append(
            f"2. Coverage: {n['n_scanned']} of {n['n_total']} candidate "
            f"configurations scanned ({n['n_skipped_budget']} skipped by budget, "
            f"{n['n_failed']} failed, {n['n_invalid']} invalid)."
        )
    else:
        lines.append("2. Coverage: not recorded in this bundle.")

    stages: list[str] = []
    for design in designs:
        for stage in _VALIDATION_STAGES.get(design, ()):
            if stage not in stages:
                stages.append(stage)
    lines.append(
        "3. Validation battery: "
        + ("; ".join(stages) if stages else "none recorded")
        + "."
    )

    present: set[str] = set()
    for cfg in pool:
        effects = (cfg.get("summary") or {}).get("effects")
        if isinstance(effects, dict):
            present.update(k for k, v in effects.items() if isinstance(v, dict))
    estimators = [label for key, label in _ESTIMATOR_LABELS if key in present]
    lines.append(
        "4. Effect estimators: "
        + (", ".join(estimators) if estimators
           else "no effect estimates (no outcome column was provided)")
        + "."
    )

    figures = [
        f["name"] for f in (r.get("figures") or [])
        if isinstance(f, dict) and f.get("name")
    ]
    lines.append(
        "5. Figures: " + (", ".join(figures) if figures else "none rendered") + "."
    )
    return "\n".join(lines)


def generate_method_diagram(bundle: ResultsBundle, out: str | Path) -> Path:
    """Render the pipeline diagram for ``bundle`` to ``out`` via paperbanana.

    Lazy import: a missing library raises ImportError naming the extra
    (``pip install "natex-discovery[paperbanana]"``). Documented single call
    contract: ``paperbanana.generate_diagram(description=..., output_path=
    str(out))``; returns ``Path(result or out)`` (the library's returned path
    when it gives one, else ``out``).
    """
    try:
        import paperbanana
    except ImportError as exc:
        raise ImportError(_PAPERBANANA_EXTRA_MSG) from exc
    result = paperbanana.generate_diagram(
        description=_pipeline_description(bundle), output_path=str(out)
    )
    return Path(result or out)
