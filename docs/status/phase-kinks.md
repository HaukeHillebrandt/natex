# Kink-design extension status — RKD and difference-in-kinks

## What shipped

- Known-cutoff sharp/fuzzy RKD and sharp/fuzzy DiK local-polynomial estimators.
- Explicit right-minus-left sign convention and post-minus-pre DiK contrast.
- HC1 and unit-clustered CR1 covariance, fuzzy cross-equation covariance, first-stage F,
  weak-design flag, delta-method interval, and honest Fieller set shapes.
- Seeded RKD/DiK DGPs with analytic bias oracles, including the stable nuisance kink that
  biases a cross-sectional RKD but cancels in DiK.
- `natex kink` with strict sharp/fuzzy arguments and NaN-clean `kink.json` output.
- Public Python exports and a method card tied to IZA DP 18313.

## Deliberate boundaries

- The workflow evaluates a user-specified, known-cutoff design. It does not search unknown
  cutoffs or reform dates; doing so needs max-over-search calibration or an honest split.
- The paper has no automatic DiK bandwidth selector. The CLI requires a bandwidth and the
  method card requires sensitivity analysis.
- The conventional interval is not robust-bias-corrected.
- A coefficient event study, joint pretrend test, automated bandwidth/donut/placebo grids,
  density-change test, and report/paper bundle integration remain follow-ups.
- Fuzzy DiK's positive-weight interpretation is documented with an additional stable latent
  composition (or valid reweighting) assumption because the paper's proof otherwise changes
  measures between periods.

## Verification

Targeted kink tests:

```bash
uv run pytest -q tests/test_kink.py tests/test_synthetic_kink.py tests/test_cli_kink.py tests/test_kink_docs.py
```

Final repository gates:

```bash
uv run ruff check src tests
uv run pytest -q
```

Collection after this phase: 955 non-backtest tests plus 32 opt-in real-data backtests.
The targeted command above runs 43 kink tests.
