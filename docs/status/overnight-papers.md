# Overnight paper sprint — status of record (2026-07-18)

Ten mini-papers plus a capstone survey, written overnight from the completed
analysis runs (`~/dev/study-runs/<slug>/`, `~/dev/epoch-data/{natex-runs,kink-runs}/`),
committed one paper per commit, and published with the flagship on the Pages site.

- **Site index:** <https://haukehillebrandt.github.io/natex/>
- Build: `scripts/build_papers.sh` via `.github/workflows/paper.yml`
  (tectonic PDF + LaTeXML HTML per paper, auto-generated collection index).
- Every figure script asserts its paper's numbers of record against the frozen
  analysis outputs before drawing; refusals and degenerate auto-configurations
  are reported verbatim ("stated as run").

## The ranked studies (capstone ordering: credibility x interest)

| # | Study | Paper | Verdict | Headline |
|---|-------|-------|---------|----------|
| 1 | US export controls vs China's compute (3 legs) | [export-controls-three-leg](https://haukehillebrandt.github.io/natex/export-controls-three-leg/) | credible (legal leg) / attenuated (total) / descriptive (big-run DiD) | legal chip stock −0.56 ln/yr (t=−3.30, 25/25 specs; Oct-2022 loophole placebo null); total ≈ half & fragile (smuggling); +3.2 runs/qtr fails placebo-in-space p=1.0 |
| 2 | BTOS AI-question rewording (measurement RDiT) | [btos-rewording-rdd](https://haukehillebrandt.github.io/natex/btos-rewording-rdd/) | credible | +6.04 pp (HAC z=16.3), 4.7x max of 49 placebos, blind LoRD3 re-localizes the cutoff; splice factor +6.04 pp / ratio 1.553 |
| 3 | EU AI Act 1e25-FLOP bunching | [euact-bunching-writeup](https://haukehillebrandt.github.io/natex/euact-bunching-writeup/) | credible (moderate) | Fisher OR 0.173 (p=0.007), 82.7% missing mass; behavioral or disclosure response — not separable |
| 4 | GPQA-Diamond kink at o1-preview | flagship, [paper](https://haukehillebrandt.github.io/natex/paper/) §GPQA | credible, date-localized | +0.00258 logit/day (t=3.32), placebo size 0/7 |
| 5 | US adoption at DeepSeek-R1 (BTOS sectors) | [btos-sector-did-r1](https://haukehillebrandt.github.io/natex/btos-sector-did-r1/) | descriptive-only | acceleration real (scan localizes first post-R1 wave, DiK +7.3 pp/yr) but placebo dates as large, 2/6 specs, crowded week (diffusion rule, Stargate, panel rollover) |
| 6 | GPT-4o sycophancy episode (LMArena) | [lmarena-sycophancy-aba](https://haukehillebrandt.github.io/natex/lmarena-sycophancy-aba/) | credible null / identified artifact | deploy style break real (perm p=0.013); BT vote gain +0.02 log-odds (se 0.14) = null; Apr-29 rollback contaminated by an arena-side serving change (pinned control moves) |
| 7 | Hyperscaler capex at ChatGPT | [capex-dik-chatgpt](https://haukehillebrandt.github.io/natex/capex-dik-chatgpt/) | identified inference artifact | +0.140 dex/yr, nominal p=4e-6 but placebo-calibrated p=0.125; pre-period empirical size 0.47 at nominal 0.05 |
| 8 | ECI at o1, fresh vintage | [eci-fresh-kink-o1](https://haukehillebrandt.github.io/natex/eci-fresh-kink-o1/) | null with power | t=0.50 at the cutoff, placebos reject 4/8 elsewhere; CI-weighted "kink" is smooth concavity, not o1 |

Supporting mini-papers (in the capstone's narrative, not its findings table):

| Study | Paper | Role |
|-------|-------|------|
| Prop-99 blind validation | [prop99-validation-writeup](https://haukehillebrandt.github.io/natex/prop99-validation-writeup/) | validation anchor: all 3 scan methods recover (California, 1989) exactly; ATT −19.5 matches ADH |
| Semiconductor event studies | [semis-event-studies-writeup](https://haukehillebrandt.github.io/natex/semis-event-studies-writeup/) | rally-not-rule: nominal kinks at 3 cutoffs killed by the battery |
| Chinchilla adoption ramp | [chinchilla-writeup](https://haukehillebrandt.github.io/natex/chinchilla-writeup/) | honest miss: diffusion ramps defeat level-break scans; documented as method gap |
| Capstone survey | [capstone](https://haukehillebrandt.github.io/natex/capstone/) | 17 analyses to verdict: 4 credible, 2 powered nulls, ≥6 nominally significant results killed by the battery |

## Excluded candidates (reasons of record — capstone Table "Considered and excluded", 15 pairs)

Highlights; full table in `papers/capstone/main.tex`:

- DCM inference-price kink at DeepSeek-V3 — series ends 2025-02 with five flat terminal months; no post-slope.
- Epoch cyber-vulnerabilities event study — no processed dataset (download is the raw CVE git repo).
- BTOS state panel Bartik DiD — needs external exposure weights; ~40% of cells disclosure-suppressed.
- R&D input-share table — n=7 rows; no design applies.
- GPQA/METR o1 kinks, China DiK as standalone minis — already results of record in the flagship; duplication.
- GPU-cluster/datacenter cumulative-growth kink — already in the flagship graveyard (placebo empirical size 1.0).
- Chip-level SuDDDS at the Apr-2025 H20 round — strongest queued follow-up (needs chip-alias map).
- Datacenter site-level SuDDDS at Colossus/Stargate — queued (18% unflagged projections).
- LMArena leaderboard RKD; style-premium DiD at spec updates — cumulative-series precedent / judge validation not yet run.
- Revenue/WAU kink at ChatGPT — zero pre-cutoff mass; not formalizable.
- Polling threshold RDs; CCAI values — single snapshot, no microdata / no treatment channel.
- Benchmark IRT teaching-to-the-test DEE — queued (needs 52-benchmark IRT panel).
- Bunching at 1e26 FLOP (US EO line) — only 3 models ever above the line.

## natex bugs filed from this batch (same-day rule)

Filed during tonight's paper pass (from the analyses' caveats):

- [#51](https://github.com/HaukeHillebrandt/natex/issues/51) bug/high — kink/DiK HC1+CR1 badly oversized on serially correlated aggregates (capex: empirical size 0.47 at nominal 0.05); no HAC option, no autocorr warning, no built-in placebo-calibrated p.
- [#52](https://github.com/HaukeHillebrandt/natex/issues/52) bug/medium — survey role auto-assignment yields degenerate/mechanical "credible" families (outcome=run counter, outcome=sector, treatment=post rediscovery; 3 studies).
- [#53](https://github.com/HaukeHillebrandt/natex/issues/53) enhancement — no first-class group-DiK; group-as-time aliasing hack used in 2 papers, output cells mislabeled.
- [#54](https://github.com/HaukeHillebrandt/natex/issues/54) enhancement — kink estimator has no per-observation weights; ECI CI-weighting required a shadow local-WLS validated against natex (68 cells, tol 1e-8).

Filed earlier the same day from the analysis pass: [#48](https://github.com/HaukeHillebrandt/natex/issues/48) (repeated `--cutoff` flags silently collapse), [#49](https://github.com/HaukeHillebrandt/natex/issues/49) (did/survey intake ignores declared `--time`, can pick a string date column), [#50](https://github.com/HaukeHillebrandt/natex/issues/50) (sc runner: multi-unit treated indicator; single-ticker panel error message).

The string-dim `.item()` DiD crash re-hit in panel construction was already fixed (`ea0c7ac`); the workaround notes in the study READMEs predate the fix.

## Wave 2 (2026-07-19)

Six new mini-papers from the wave-2 scout/analysis pass, one commit per paper,
wired into the collection (`papers/README.md` rows 13–18, cover contents,
`ALL_PAPERS.pdf` rebuilt to 116 pages / 19 bookmarks) in `dc69db0`. Same rules
of record: seeded runs, figure scripts hard-assert the numbers of record,
refusals stated as run.

### The ranked wave-2 studies

| # | Study | Paper | Verdict | Headline |
|---|-------|-------|---------|----------|
| 13 | Chip-level export controls (SuDDDS rematch) | [chip-suddds-exportcontrols](https://haukehillebrandt.github.io/natex/chip-suddds-exportcontrols/) | date-localization positive (Oct-23) / unattributable magnitude (H20) | all 6 known-treatment scans put max-LLR at the declared quarter exactly; Oct-2023 A800/H800 ban significant (Bernoulli p=0.030, shipments to literal zero in one quarter); H20 tau ≈ −2.8 asinh (CR1 t=−4.23) but lifecycle churn generates same-size placebos — donor pool contaminated (3/6 Chinese substitutes) or too small (3 clean, p=0.75) |
| 14 | BTOS spliced panel: R1 kink out of sample | [btos-spliced-r1-extension](https://haukehillebrandt.github.io/natex/btos-spliced-r1-extension/) | honest reversal | prior +7.3 pp/yr group DiK replicates exactly on its old window but uses zero new data; every spec seeing the extended panel is null-to-negative at R1 (bw 1.0: −0.16, p=0.96); two-sided placebo battery (positive kinks at 2024 non-events, negative at 2025H2 + splice boundary) — convex-then-plateau diffusion gap explains all cutoffs; gap slope decelerated +4.4 → +2.3 pp/yr |
| 15 | Benchmark contamination, public vs held-out (DEE) | [benchmark-contamination-dee](https://haukehillebrandt.github.io/natex/benchmark-contamination-dee/) | suggestive-sign null | +0.127 z post × public premium fails WCB p=0.145 / exact RI p=0.213, collapses under IRT-difficulty controls (+0.04) and model FE (+0.03), and fails placebo-cutoff localization (−0.5 yr placebo +0.267 > true cutoff); held-out post shortfall −0.19 z sign-matches the story; DiK secondary is a correctly-identified artifact |
| 16 | Datacenter sites at 2025Q1 (Stargate quarter) | [datacenter-sites-2025q1](https://haukehillebrandt.github.io/natex/datacenter-sites-2025q1/) | identified artifact + clean null | fivefold additions "break" is smooth ~40%/q exponential growth (2025Q1 never best on the 6-date placebo grid, rank 3–6; log-kink negative everywhere); scan's t0=2025Q1 p=0.010 mechanically recovers the constructed 21/73-site coding (role-assignment triage); neocloud attribution null: tau=+0.098 asinh-MW, t=0.699, permutation p=0.56 |
| 17 | State AI-exposure gradient at R1 (AEI × BTOS) | [aei-btos-state-gradient](https://haukehillebrandt.github.io/natex/aei-btos-state-gradient/) | identified artifact ("manufactured gradients") | +1.635 pp/yr per AEI unit, permutation p=0.257; placebo cutoffs bracket it (+1.56, +2.24) and the same-day RTO remote-work placebo outcome matches it (+1.97, weighted version nominally significant p=0.045); TWFE level effect (+0.527 pp, p=0.011) is a pre-trend that flips sign under state trends; long-window negative gradients are a rewording-splice artifact (ratio misprices high-AEI states, corr +0.676) |
| 18 | CVE publications at R1 | [cve-monthly-kink-r1](https://haukehillebrandt.github.io/natex/cve-monthly-kink-r1/) | calibrated null + localized late-2025 surge | R1 kink +0.038 ln/yr (z=0.30); scout's +0.18 is a uniform-kernel far-from-cutoff artifact (placebo-calibrated p=0.80, empirical size 0.14 = issue #51); blind LoRD3 ranks Nov/Dec-2025 (scan p=0.02/0.01) and Feb–Apr-2026 first, R1 in no top-20; surge arrives 10–17 months post-release — NVD backlog catch-up + CNA expansion, not AI attribution |

### Considered and rejected in wave 2 (reasons of record)

- DCM inference-price kink at DeepSeek-V3 (extension leg) — verified infeasible:
  every price table on disk ends 2025-01/02 (same terminal date as the processed
  series); no post-V3 slope data without a fresh Artificial-Analysis-class scrape.
- FLOP-disclosure-collapse kink at GPT-4 (disclosure-as-outcome) — disclosure
  share is a gradual ramp (0.85 → 0.42, 2022Q3–2024Q4); the only sharp break
  (2025Q4) is a right-edge Epoch estimation-lag artifact — would repeat the
  Chinchilla failure mode.
- AEI occupation/task time-series panel — current schema covers only Apr–May
  2026 (2 months); earlier releases (v1–v6) methodologically incompatible; only
  the cross-sectional exposure use survives (study 17).
- Bunching at 1e26 FLOP (US EO line) — still only 3 models ever above the line.
- BTOS state Bartik with ACS/BLS external weights — superseded by the
  AEI-exposure version (study 17); doesn't fix suppressed small-state cells.
- LMArena leaderboard RKD / style-premium DiD at spec updates — no new data;
  cumulative-series precedent (placebo empirical size 1.0) and judge validation
  unaddressed.
- Chinchilla hygiene rematch (drop fine-tune rows, rescan) — diagnosis
  confirmation for a published honest miss, not a new finding; low novelty
  under the study cap.
- Sales-panel Oct-2023 unsupervised-discovery effect leg as standalone — naive
  placebo-in-space fails (p=0.73); folded into study 13 instead.

### natex gotchas filed from this batch (same-day rule)

- [#55](https://github.com/HaukeHillebrandt/natex/issues/55) enhancement —
  effect-leg placebo inference refuses ("only 0 usable placebos; >= 5
  required") on unbalanced/short/single-binary panels with no built-in
  fallback; 4 of 6 wave-2 studies hand-rolled shadow TWFE/permutation/WCB legs.
- Already on file, re-hit this wave (no dupes filed): #51 (HC1/CR1 oversized on
  smooth serially correlated aggregates — cve, aei-btos, chip placebo flavors),
  #53 (group-DiK aliasing hack — btos-spliced-r1-extension reuses it).
