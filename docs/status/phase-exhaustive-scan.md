# Status — exhaustive scan, inconclusive verdicts, calibrated calendar-time inference (v0.3.0)

**Date:** 2026-10-02. **Branch:** `fix/survey-inference-and-exhaustive-scan`.
**Trigger:** the review of the v0.2.0 paper collection (18 papers) and the question
whether the automated pipeline still applies the LoRD3 scan exhaustively or lets the
analyst layer pick one or two plausible designs.

## Diagnosis of record

- LoRD3 was exhaustive *within* a configuration (every center's k-neighborhood, every
  hyperplane through it). The configuration list was not: `enumerate_configs` derived
  one rdd and one did candidate from the bound spec, so the search plan — null-backend
  heuristics or an LLM — set the role space. The null backend's outcome guess (the first
  numeric column) was reserved from the forcing space, which on the test fixture removed
  the planted running variable from every scan.
- The survey reported five "could not test" outcomes as `null`: a refused sc placebo
  test, no scannable configuration, a mechanical time-step rediscovery, degenerate
  kink/density fits, and a degenerate DEE ensemble.
- The survey's kink family gated calendar-time cutoffs on Holm-adjusted HC1 p-values
  although issue #51 had measured that inference as badly oversized and the kink method
  card already named placebo calibration as the inference of record.
- Elapsed-time running variables (`days`, `days_since_o1`, `wave`, `t`) were not
  recognized as calendar time, so the previous flagship's own series would have taken the
  nominal path.

## What shipped

Release notes: [docs/release_notes/v0.3.0.md](../release_notes/v0.3.0.md). In short:
`discover(profile=, outcome_candidates=, known_outcomes=)` with `enumerate_role_configs`
and recorded exclusions; `Dataset.with_outcome` and `effects_by_outcome`; `best()` by
scan p then LLR; the `inconclusive` verdict across all seven family runners and the
report badge; the placebo-calibrated gate with `MIN_PLACEBO_CUTOFFS = 19` and a
49-position grid; `PlaceboKinkGrid.min_attainable_p`; the profiler's elapsed-time name
rule; method cards, README, AGENTS.md and the three skills updated, including a style
contract for generated papers.

## Run of record

- `uv run pytest -q`: **1269 passed, 32 backtests deselected** (209.8 s, Python 3.13).
- `uv run ruff check src tests`: clean.
- Real-data checks on the previous flagship's frozen series
  (`~/dev/epoch-data/kink-runs/`, not committed):

| Series | Spec | Nominal | Placebo grid | Calibrated p | Verdict under the new rule |
|---|---|---|---|---|---|
| GPQA-Diamond logit score at o1-preview | bw 540 d, triangular | z = 3.32 | 49 evaluable, 7 reject (size 0.14) | 0.020 (floor) | most extreme of 49 placebo cutoffs |
| same | bw 365 / 730 d | z = 2.96 / 4.05 | 49 / 46 evaluable | 0.020 / 0.021 | most extreme at every bandwidth |
| same, `natex survey` default bw 244.5 d | median \|days\| | p = 0.069 | 49 evaluable | 0.06 | null |
| China legal chip stock (15 quarters) | bw 548 d | z = −5.09 | 8 positions, 7 evaluable | floor 0.125 | not gateable at 5% (< 19 positions) |
| Hyperscaler chip stock (15 quarters) | bw 548 d | z = −3.48 | 7 evaluable | floor 0.125 | not gateable |

The GPQA result therefore survives placebo calibration once a large enough grid is used;
the v0.2.0 paper's 7-placebo grid could not have shown that (its floor was 0.125). The
quarterly chip-stock series cannot host enough placebo positions for a calibrated 5%
claim on any bandwidth; sign stability across specifications is the only evidence
available there, and the paper should say so.

## Deliberately not done here

- The papers under `paper/` and `papers/` are untouched: re-running them under the new
  rules needs the frozen data extracts and is an editorial decision (which papers to keep)
  recorded in the review, not a pipeline change.
- did effects are still estimated for the configuration's single outcome; per-outcome
  effects for did would need a panel rebuild per outcome.
- The `>= 5 usable placebos` refusal in the DiD effect leg keeps its opt-in permutation
  fallback (issue #55); the survey still runs the default refusal, now reported as
  `inconclusive` rather than `null`.
- The polling dataset in `~/dev/epoch-data/extracted/` is a 59-row crosstab of
  percentages without respondent microdata, so the age-65 retirement discontinuity
  remembered from an early run cannot be reproduced from it; that run used other data.
