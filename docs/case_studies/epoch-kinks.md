# Case study: kink designs on Epoch AI datasets

> Paper: [HTML](https://haukehillebrandt.github.io/natex/paper/)
> ([PDF](https://haukehillebrandt.github.io/natex/main.pdf)), source in
> [`paper/`](../../paper/). This file is the numbers of record.

Point estimates: `natex kink` v0.2.0, analysis pass of 2026-07-16. Placebo grids and
calibrated p-values: natex v0.3.0 on the same extracts, 2026-10-05. Calendar-time runs use
a unit policy denominator, so the kink is a descriptive slope change per day. The
inference of record is the placebo-calibrated p (add-one rank of the declared date's |z|
among up to 49 shifted placebo cutoffs); nominal HC1 p-values are reported only as a size
diagnostic. Data: Epoch AI, CC-BY 4.0, not committed.

## Headline table

| Series | Cutoff | bw (d) | kink / day | nominal z | placebos | nominal rejection rate | calibrated p | verdict |
|---|---|---|---|---|---|---|---|---|
| GPQA-Diamond, logit(mean score), 180 models | o1-preview 2024-09-12 | 540 | +0.00258 (se 0.00078) | 3.32 | 49 | 14% | 0.020 | dated bend |
| METR 50% time horizon, log2 min, 48 models | o1-preview | 720 | +0.00601 (se 0.00282) | 2.13 | 17 | 24% | 0.278 | era bend, not dated |
| Epoch Capabilities Index, all 455 models | o1-preview | 540 | -0.00725 (se 0.00895) | -0.81 | 49 | 55% | 0.780 | null |
| China legal vs hyperscaler ln H100e stock, group DiK | controls 2023-10-17 | 548 | -0.00154 (se 0.00047) | -3.30 | 7 | n/a | 0.125 (floor) | inconclusive |

## Bandwidth grid (per-model and monthly series)

| Series | bw | kink / day | z | placebos | nominal rejection rate | calibrated p |
|---|---|---|---|---|---|---|
| GPQA | 365 | +0.00313 | 2.96 | 49 | 0% | 0.020 |
| GPQA | 540 | +0.00258 | 3.32 | 49 | 14% | 0.020 |
| GPQA | 730 | +0.00261 | 4.05 | 46 | 4% | 0.021 |
| METR 50% | 540 | +0.00507 | 1.41 | 19 | 16% | 0.350 |
| METR 50% | 720 | +0.00601 | 2.13 | 17 | 24% | 0.278 |
| METR 50% | 900 | +0.00627 | 2.37 | 14 | 29% | 0.200 |
| ECI | 365 | -0.01190 | -0.72 | 49 | 39% | 0.660 |
| ECI | 540 | -0.00725 | -0.81 | 49 | 55% | 0.780 |
| ECI | 730 | -0.00463 | -0.70 | 49 | 67% | 0.820 |
| GPU clusters at ChatGPT (graveyard) | 708 | +0.00080 | 8.95 | 49 | 88% | 0.620 |

## China group DiK (quarterly; 15 quarters per group)

| Contrast vs hyperscalers | bw | tau / day | t | positions / evaluable | calibrated p |
|---|---|---|---|---|---|
| China legal | 548 | -0.00154 | -3.30 | 10 / 7 | 0.125 (floor) |
| China legal | 730 | -0.00114 | -2.85 | 9 / 6 | 0.143 (floor) |
| China total incl. smuggled | 548 | -0.00076 | -1.54 | 10 / 7 | 0.125 |
| China total incl. smuggled | 730 | -0.00046 | -1.10 | 9 / 6 | 0.429 |
| Other buyers (placebo-treated group) | 548 | +0.00050 | 1.34 | 10 / 7 | 0.750 |
| China legal vs neocloud controls | 548 | -0.00094 | -1.70 | 10 / 7 | 0.375 |
| China legal, Oct-2022 round (-375 d) | 548 | +0.00182 | 1.46 | | |

The sign is negative in all 25 specifications of the July pass and the two falsifications
behave (placebo group null, loophole round null). Quarterly data cannot host more than
seven placebo positions, so no calibrated claim below p = 0.125 is possible: the result
is sign-robust and inconclusive under the inference of record. The estimator's
autocorrelation warning fires on most of these cells.

## Notes per series

- **GPQA.** Slopes 0.00188 -> 0.00446 logit/day (about 15 -> 34 raw pp/yr). Positive in
  8/8 bandwidth-donut cells; release-density kink null (t = 0.76); compute-covariate kink
  null (p = 0.54). Largest placebo |z| at bw 540: 2.78 at -142 days. Composition
  (reasoning models entering the release stream) is the mechanism.
- **METR.** Slopes 0.00331 -> 0.00932 log2/day (doubling 9.9 -> 3.5 months). A placebo at
  -311 days gives |z| = 7.9; placebos at -196 and +455 days also exceed the declared
  date. The 80% horizon has 3 pre-cutoff models and 9 placebo positions: not estimable.
- **ECI.** About 18 points/yr on both sides. 38 of 49 placebo dates give a larger
  statistic. The nominal rejection rate (55%) is a size diagnostic; the July text's
  "null with power" reading of it is withdrawn.
- **Graveyard.** GPU clusters at ChatGPT: curvature, calibrated p 0.62. Chinchilla:
  sign-unstable null (t = -0.91, -0.05, +1.48 at bw 365/540/730). EU AI Act 1e25 line: a
  level notch with sorting (Fisher OR 0.17, p = 0.007), analysed as bunching in the
  companion note.

## What changed from the July version

- Placebo grids went from 7 positions (floor 0.125) to up to 49 (floor 0.02).
- GPQA: "credible, 0/7 placebos reject" became "most extreme of 49, calibrated p 0.02".
- China: "credible with an honesty band" became "sign-robust, inconclusive (7 positions)".
- ECI: the "test has power because placebos reject" claim was dropped; rejections
  elsewhere measure the nominal test's size.
- Nominal HC1 p-values are no longer used for any verdict.

Reproduction: `natex kink` at the bandwidths above, then `placebo_kinks` on the grid from
`natex.survey.runner._placebo_cutoff_grid` and `placebo_calibrated_p`; see
`docs/method_cards/kink.md`.
