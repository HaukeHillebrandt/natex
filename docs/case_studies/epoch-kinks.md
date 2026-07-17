# Case study — kink designs on Epoch AI datasets (July 2026 field pass)

Four formal results from running `natex kink` (v0.2.0) over Epoch AI's open datasets
(CC-BY 4.0, [epoch.ai/data](https://epoch.ai/data)). Two calendar-time RKDs, one
falsification companion, one group difference-in-kinks. All candidates had externally
dated cutoffs — none was searched for. Calendar-time runs use a dummy unit denominator
(`--policy-kink 1.0`), so tau is a descriptive slope change (outcome units/day), not a
marginal causal response; see the method card's
[Time as the running variable](../method_cards/kink.md#time-as-the-running-variable).
**No data files are committed here** — sources below are public.

| # | Series (source) | Design / cutoff | Headline (HC1) | Verdict |
|---|---|---|---|---|
| 1 | GPQA-diamond, logit(mean score), 180 models | sharp RKD-in-time at o1-preview, 2024-09-12 | +0.00258/day, se 0.00078, t = 3.32 (bw540 tri) | **Credible kink, date-localized** |
| 2 | METR 50% time horizon, log2 minutes, 48 models | sharp RKD-in-time at o1-preview, 2024-09-12 | +0.00601/day, se 0.00282, t = 2.13 (bw720 tri) | **Credible era bend, date attribution fails** |
| 3 | Epoch Capabilities Index, all models | sharp RKD-in-time at o1-preview (falsification) | −0.00725 pts/day, se 0.00895, t = −0.81 (bw540 tri) | **Clean null — the guard the other two need** |
| 4 | China vs hyperscaler ln cumulative H100e stock | sharp group-DiK at export controls, 2023-10-17 | −0.00154/day, se 0.00047, t = −3.30 (bw548 tri) | **Credible kink with a magnitude honesty band** |

## 1. GPQA-diamond at o1-preview — the cleanest result

Source: Epoch Benchmarking Hub, `gpqa_diamond` (180 dated models, 45 pre / 135 post;
sample starts 2023-03-14). Primary outcome logit(mean score) — the score ceiling bends raw
slopes mechanically. Slope 0.00188 → 0.00446 logit/day (raw: **15 → 34 pp/yr, ~2.4x**).
Diagnostics: positive in 8/8 bandwidth-donut cells (t 2.15–3.71, donut *strengthens* it);
placebo-kink grid empirical size **0/7**; release-date density kink null (t = 0.76);
compute-covariate kink null (p = 0.54). Degree-2 keeps sign and magnitude at 3x the SE.
Standing caveat: composition — reasoning models entering the release stream — *is* the
mechanism, so tau is a release-stream property, not per-model improvement.

## 2. METR 50% time horizon at o1-preview — bend exists, date does not

Source: `metr_time_horizons_external` via Epoch's benchmark data (METR's long-task suite;
12 pre / 36 post models). Slope 0.00331 → 0.00932 log2/day at bw720 tri: doubling time
**9.9 → 3.5 months** (full-sample uniform: +0.00491, se 0.00095, t = 5.17; 7.6 → 3.6 mo).
Positive in 8/8 bandwidth-donut cells (0.0044–0.0085); 80%-horizon variant agrees
(+0.00454, t = 4.08, only 3 pre points — directional only). But the **pre-side placebo
grid smears**: at bw720, shifted cutoffs at −270/−180/−90 days all reject (p = 0.030,
0.014, 0.004) with estimates the size of the headline; post-side shifts are clean nulls;
empirical size 3/6. With 11–12 pre points, adjacent windows share most of their data.
Verdict language used: *"credible slope change with failed date-localization — report as
reasoning-era slope doubling, not 'o1-preview caused X'."* This is the era-bend vs
event-bend contrast with case 1.

## 3. Epoch Capabilities Index — the sibling-series falsification

Source: `epoch_capabilities_index` (IRT-linked aggregate; 356 dated, scored models). Run
*expecting* null, as the guard against "everything kinked in 2024": all-models kink null in
8/8 grid cells (t −0.60 to −1.54; ~18 ECI pts/yr on both sides), while the placebo grid
rejects at **4/7 shifted cutoffs** — the test has power in this data and still reads zero
at the o1 date. So the per-benchmark bends (1, 2) are not a global measurement shift.
Two recorded lessons: (a) an analyst-script frontier filter
(`cummax().diff().fillna(1) > 0` over 231 NaN scores) manufactured 347 false "records" —
the clean frontier has 25 points and is not an estimable design (4 left-cell points; fails
donut, bandwidth-direction, and placebo checks in every direction; never quote it);
(b) a side finding, compute-covariate kink −0.00265 log10/day at o1 (p = 0.021),
corroborates the "pretraining-compute frontier stalls" story without being a design.

## 4. China chip stock at the Oct-2023 export controls — group-DiK

Source: `ai_chip_owners/cumulative_by_designer` (quarterly ln cumulative H100e; the
`Incomplete` 2026-03-31 quarter, whose cumulative totals *decrease*, must be dropped).
Böckerman–Jysmä–Kanninen DiK with periods aliased to **groups**: treated = China,
control = hyperscalers (Amazon+Microsoft+Google+Meta), same calendar cutoff; the controls
difference out the global supply bend (their own kink −0.00067/day). Legal-stock DiK
−0.00154/day ≈ **−0.56 ln-units/yr** growth-rate change (CR1 by year: t = −5.15, G = 4 —
few-cluster caveat); negative in **25/25 specifications**. Falsifications: the Oct-2022
first-round placebo is null/positive (+0.0018, p = 0.15 — the A800/H800 loophole round);
the placebo-treated group ("Other" vs hyperscalers) is clean at bw 548–730 but rejects at
bw365 → defensible range bw548–730, DiK −0.0009 to −0.0015. **Total stock incl. smuggled
is roughly half and fragile** (−0.00076, t = −1.54 at bw548): the smuggled series (starts
2024-03-31, itself a treatment response) substitutes for lost legal supply. Serial
dependence in a 6-points-per-cell cumulative series makes |t| optimistic; the sign
stability is the real evidence. Verdict language used: *"the policy bent the legal channel
≈ −0.3 to −0.56 ln-units/yr; the effect on China's actual compute stock is smaller and
fragile."*

## What transferred back into natex docs

- The joint-cell HC1 dof convention (METR cross-check: kink identical to 10 decimals,
  0.0067291763; SE 0.0023236 vs 0.0024334 per-side) — now in the method card's
  [SE convention](../method_cards/kink.md#se-convention).
- The placebo grid read as bend-existence vs date-attribution, and the sibling-series
  falsification pattern — now in
  [Time as the running variable](../method_cards/kink.md#time-as-the-running-variable).
- Rejections are results: a GPU-cluster "kink" at ChatGPT with t = 12.9 was discarded
  because its placebo grid rejected at **7/7** shifted cutoffs (empirical size 1.0) —
  smooth super-exponential curvature bends everywhere. The EU AI Act threshold was
  rejected as a *step* (level, not slope) with established bunching at 1e25 FLOP, which
  violates any no-sorting requirement; the bunching test remains the right tool there.

Reproduction: Epoch datasets above, `natex kink` CLI with the bandwidths/kernels listed,
plus the `sensitivity_grid` / `placebo_kinks` diagnostics battery from the method card.
