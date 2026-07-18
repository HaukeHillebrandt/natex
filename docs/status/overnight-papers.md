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
