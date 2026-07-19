> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/btos-sector-did-r1/) · [PDF in this repo](./main.pdf)

# Introduction

On 20 January 2025 the Chinese lab DeepSeek released R1, an openly licensed reasoning model trained at a reported cost far below frontier-lab budgets . The release was treated by markets as news about the cost curve of AI: on 27 January 2025 Nvidia alone lost roughly \$590 billion of market capitalization, the largest one-day loss on record, and an event-study literature on the “DeepSeek shock” emerged almost immediately.[^2] A natural real-economy question follows: did cheap, open reasoning models bend the *adoption* curve — the share of US businesses actually using AI in production?

Measurement of that adoption curve is recent. The Census Bureau’s Business Trends and Outlook Survey (BTOS) has asked a probability sample of US firms since September 2023 whether they used AI in producing goods or services in the last two weeks; introduce these data and document a rise in the national use rate from 3.7% to 5.4% over the first six months, concentrated in the Information sector and larger firms. Complementary measurement includes the 2018 Annual Business Survey module of (pre-LLM adoption below 6% of firms, over 18% employment-weighted) and the multi-country executive survey of (high headline adoption, small realized effects). This literature is descriptive by design; formal tests of whether adoption *trends* broke at specific AI events are, to our knowledge, absent.

A trend break for one group relative to another at a known date is the estimand of the difference-in-kinks (DiK) design formalized by , building on the regression kink design of . Kink estimates in time-series settings are known to produce spuriously significant slope changes, which is why placebo distributions at non-event dates are the appropriate yardstick . This paper applies the `natex` toolkit’s DiK and scan-based DiD estimators to the BTOS sector panel with DeepSeek-R1 as the candidate cutoff, o1 (12 September 2024) as a secondary contrast, and a mandatory falsification battery, which converts an apparently significant event-study result into an explicit non-attribution.

# Data

The primary source is the Census Bureau’s BTOS AI core workbook (DRB approval CBDRB-FY25-0425): sector-level biweekly estimates and standard errors for Question 7, “In the last two weeks, did this business use Artificial Intelligence (AI) in producing goods or services?”, answer “Yes,” in percentage points. The panel covers 20 sectors (19 two-digit NAICS sectors plus an unclassified residual) over 54 biweekly waves, 202319–202520 (collection periods 2023-09-11 through 2025-10-05); 199 of 1,080 sector-waves are disclosure-suppressed, leaving 881 usable cells. The panel predates the 2025-12-03 questionnaire rewording entirely. The running variable $`t`$ is the fractional year of each wave’s collection-period midpoint. The exposed group is {51 information, 52 finance and insurance, 54 professional/scientific/technical services}; the kink-leg comparison group is {11, 21, 23, 72}. A placebo outcome (the archived remote-work Question 6 “Yes” share) comes from the BTOS archive workbook (CBDRB-FY23-0478/FY24-0474/FY25-0117), waves 202417–202514.

Coverage is the binding data limitation: sectors 11, 21, 22, and 55 are almost entirely suppressed (3, 3, 4, and 8 usable waves of 54), so the “unexposed” comparison group is effectively sectors 23 (construction) and 72 (accommodation and food services).

# Design and Methods

#### Group difference-in-kinks.

Following , the DiK estimand is the change in the slope of the outcome in calendar time at a known cutoff for the exposed group *minus* the same slope change for the comparison group, estimated by local linear regression on each side of the cutoff within each group (`natex` v0.2.0 ). The primary specification uses cutoff $`t_0=2025.055`$ (DeepSeek-R1; first post wave 202503, collected 2025-01-27 to 02-09), bandwidth 0.5 years ($`\pm 13`$ waves), triangular kernel, a one-wave donut, and cluster-robust (CR1) standard errors clustered by sector with 7 clusters and critical values from $`t(G-1)`$. With 7 clusters and only 2–4 per DiK cell, CR1 inference is fragile even with the small-sample correction ; we therefore lean on design-based falsification rather than on the standard errors.

#### Falsification battery.

\(i\) a specification grid over bandwidth $`\in \{0.25, 0.5, 1.0\}`$ years, kernel $`\in`$ {triangular, uniform}, and donut $`\in \{0, 1\ \text{wave}\}`$; (ii) placebo cutoffs at four non-event dates (2024.30, 2024.50, 2024.85, 2025.30), in the spirit of the permutation logic of ; (iii) a placebo-in-space battery of per-sector local kinks at the R1 date; (iv) a placebo *outcome* (remote work) at the same cutoff; and (v) a secondary contrast at o1 (2024-09-12, $`t=2024.6967`$): if R1 mattered over and above the reasoning-model wave o1 began, the exposed-minus-unexposed gap should *decelerate* there.

#### Sector-by-wave DiD.

The `natex` survey pipeline’s sector-by-wave DiD scan (SuDDDS) searches over candidate break dates $`t_0`$ and window widths $`W`$, maximizing a likelihood-ratio scan statistic for a declared-treatment DiD, calibrated by permutation under an AR(1)-by-unit null . We also fit a manual two-way fixed-effects (TWFE) level DiD with wave and sector fixed effects, CR1 by sector, and a placebo-in-space randomization check.

#### Honest-inference notes, stated as run.

The pipeline’s refusals are reported, not patched over: the SuDDDS anticipation test refused (Holm $`p=[\mathrm{NaN},\mathrm{NaN},\mathrm{NaN}]`$); the dd/synthetic-control/GESS effect estimators refused with $`\tau=\mathrm{NaN}`$ (0 usable placebo units on this unbalanced panel, $`\geq 5`$ required), hence the manual TWFE estimate below; and the automated survey’s own family verdicts were discarded as role-assignment artifacts (its “credible” RDD, scan $`p=0.010`$, $`\tau_{\text{2sls}}=0.749`$, se $`0.343`$, is the mechanical rediscovery of candidate treatment $`=`$ post with outcome $`=t`$; its kink family, Holm $`p=0.807`$, tested outcome $`=`$ sector — vacuous; its DiD family, $`p=0.70`$, used a time-invariant treated indicator rather than the declared design; iv/sc/bunching returned needs-input; dee was null with 1 usable experiment).

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) BTOS AI-use share by exposure group across all 54 waves, with the o1 and DeepSeek-R1 release dates marked. (b) Group difference-in-kinks estimates (primary specification: bandwidth 0.5 yr, triangular kernel, one-wave donut) at the R1 cutoff, the o1 cutoff, and four non-event placebo dates, with 95% CR1 confidence intervals; filled markers are significant at the 5% level against <span class="math inline"><em>t</em>(<em>G</em> − 1)</span> critical values. Two non-event placebos are significant, one larger than the R1 estimate.</figcaption>
</figure>

#### The bend is real …

The SuDDDS scan localizes a break at $`t_0=2025.089`$ — exactly the first post-R1 wave (202503) — with window $`W=0.536`$ yr: max LLR $`=22.65`$, scan $`p=0.030`$ on the 20-sector panel, and LLR $`=32.70`$, $`p=0.010`$ (the $`q=99`$ permutation floor) on the 7-sector subpanel; the composition check passes ($`p=1.000`$). The SE-weighted exposed-minus-unexposed gap slope over $`\pm 13`$ waves rises from $`+4.197`$ (se $`2.089`$) to $`+13.940`$ (se $`1.649`$) pp/yr, $`\Delta = +9.74`$ (se $`2.66`$). The primary group DiK at R1 (Table <a href="#tab:dik" data-reference-type="ref" data-reference="tab:dik">1</a>, first row) is $`\hat\tau = +7.297`$ pp/yr (CR1 se $`2.797`$, 95% CI $`[0.454, 14.140]`$, $`|t|=2.61`$ against $`t(6)`$ critical value $`2.447`$, $`n=123`$): the exposed group’s slope jumps from $`3.234`$ to $`12.726`$ pp/yr against $`0.485`$ to $`2.680`$ for the comparison group. The manual TWFE level DiD gives $`\hat\tau = +4.583`$ pp (CR1 se $`0.526`$, $`t=8.72`$), and placebo-in-space is clean: 0 of 13 control-sector pseudo-treatments reach the minimum exposed $`|t|`$ (controls’ max $`7.76`$ vs. exposed $`10.19/10.49/13.19`$; exact-$`p`$ floor $`1/14 = 0.071`$), with per-sector effects of $`+4.26`$ (51), $`+4.14`$ (52), and $`+5.35`$ (54) pp.

#### … but attribution to R1 fails.

Five findings block a causal reading. *First*, the placebo-cutoff battery fails: at the non-event dates 2024.30 and 2024.50 the same DiK specification returns $`+11.419`$ (se $`3.556`$, $`|t|=3.21`$) and $`+4.023`$ (se $`0.889`$, $`|t|=4.53`$) — both significant, the first larger than the R1 estimate itself (Table <a href="#tab:dik" data-reference-type="ref" data-reference="tab:dik">1</a>, panel B; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). *Second*, the R1 estimate is significant in only 2 of 6 specification variants. *Third*, the kink is not exposure-specific: in per-sector local kinks at R1, exposed sector 52 ranks 1/16 ($`+15.41`$, $`t=6.36`$), but non-exposed sectors 71 ($`+10.46`$, $`t=4.24`$), 53 ($`+10.00`$, $`t=3.82`$), and 62 ($`+7.94`$, $`t=6.62`$) all exceed exposed sectors 51 ($`+6.56`$, $`t=1.03`$) and 54 ($`+6.50`$, $`t=3.13`$) — the January-2025 kink is broad-based. *Fourth*, the o1 deceleration contrast is bandwidth-aliased: $`-5.884`$ (se $`2.909`$, $`|t|=2.02 <`$ crit $`2.571`$, not significant, though the sign matches the prediction) at bandwidth 0.5, but $`+3.818`$ (se $`1.417`$, $`|t|=2.69`$, significant, sign *flipped*) at bandwidth 1.0; the SE-weighted gap-slope change at o1 is $`+0.59`$ (se $`2.63`$) — the expected deceleration is not reproduced. *Fifth*, the TWFE level DiD, though clean in space, violates parallel trends: the pre-R1 gap slope is $`+4.20`$ pp/yr (se $`2.09`$), so the level estimate is descriptive. The placebo *outcome* is consistent with a broad shock rather than an AI-specific one but is underpowered: the remote-work DiK at R1 is $`+8.821`$ (se $`5.351`$, $`|t|=1.65`$, not significant, on a truncated pre-window) — same sign and magnitude as the AI outcome.

<table id="tab:dik">
<caption>Group difference-in-kinks estimates, BTOS AI-use share (pp/yr).</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline">|<em>t</em>|</span></th>
<th style="text-align: center;">sig. 5%</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: DeepSeek-R1 cutoff (<span class="math inline"><em>t</em><sub>0</sub> = 2025.055</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 0.5, donut (primary)</td>
<td style="text-align: right;"><span class="math inline">+7.297</span></td>
<td style="text-align: right;"><span class="math inline">2.797</span></td>
<td style="text-align: right;"><span class="math inline">2.61</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 0.5, no donut</td>
<td style="text-align: right;"><span class="math inline">+6.742</span></td>
<td style="text-align: right;"><span class="math inline">1.188</span></td>
<td style="text-align: right;"><span class="math inline">5.68</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">uniform, bw 0.5, donut</td>
<td style="text-align: right;"><span class="math inline">+4.477</span></td>
<td style="text-align: right;"><span class="math inline">3.079</span></td>
<td style="text-align: right;"><span class="math inline">1.45</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 1.0, donut</td>
<td style="text-align: right;"><span class="math inline">+2.350</span></td>
<td style="text-align: right;"><span class="math inline">2.934</span></td>
<td style="text-align: right;"><span class="math inline">0.80</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">uniform, bw 1.0, donut</td>
<td style="text-align: right;"><span class="math inline">+3.941</span></td>
<td style="text-align: right;"><span class="math inline">2.478</span></td>
<td style="text-align: right;"><span class="math inline">1.59</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 0.25, donut</td>
<td style="text-align: right;"><span class="math inline">+10.488</span></td>
<td style="text-align: right;"><span class="math inline">8.130</span></td>
<td style="text-align: right;"><span class="math inline">1.29</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: placebo and secondary cutoffs (triangular, donut)</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.30</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">+11.419</span></td>
<td style="text-align: right;"><span class="math inline">3.556</span></td>
<td style="text-align: right;"><span class="math inline">3.21</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.50</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">+4.023</span></td>
<td style="text-align: right;"><span class="math inline">0.889</span></td>
<td style="text-align: right;"><span class="math inline">4.53</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.85</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">−1.385</span></td>
<td style="text-align: right;"><span class="math inline">3.118</span></td>
<td style="text-align: right;"><span class="math inline">0.44</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2025.30</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">−1.680</span></td>
<td style="text-align: right;"><span class="math inline">2.293</span></td>
<td style="text-align: right;"><span class="math inline">0.73</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">o1, <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span>, bw 0.5</td>
<td style="text-align: right;"><span class="math inline">−5.884</span></td>
<td style="text-align: right;"><span class="math inline">2.909</span></td>
<td style="text-align: right;"><span class="math inline">2.02</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">o1, <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span>, bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+3.818</span></td>
<td style="text-align: right;"><span class="math inline">1.417</span></td>
<td style="text-align: right;"><span class="math inline">2.69</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
</tbody>
</table>

Notes: CR1 standard errors clustered by sector; significance against $`t(G-1)`$ critical values with $`G \in \{5,6,7\}`$ clusters in the estimation window (2.447–2.776). Bandwidths in years ($`0.5\ \text{yr} = \pm 13`$ waves); donut $`=`$ one wave. Outcome: sector-level share answering Yes to BTOS Question 7, percentage points.

# Caveats and Conclusion

Three caveats are decisive. First, a calendar-time cutoff absorbs every same-week event: the Framework for Artificial Intelligence Diffusion export rule (2025-01-13) and the Stargate announcement (2025-01-21) fall in the same window as R1 (2025-01-20), so a DiK at $`t=2025.055`$ cannot distinguish among them — and the BTOS sample-year rollover (wave 202426 $`\to`$ 202501, mid-December 2024) changes respondent-panel composition three to five weeks before the cutoff. Second, the comparison group is effectively two sectors: NAICS 11, 21, 22, and 55 are almost entirely disclosure-suppressed, and CR1 inference rests on 7 clusters with only 2–4 clusters per DiK cell, where cluster-robust standard errors are known to over-reject . Third, and most important, the placebo-cutoff battery finds slope breaks as large as the R1 estimate at dates where nothing happened — exactly the failure mode placebo distributions exist to catch : on this panel, the DiK test cannot tell the R1 date apart from ordinary undulation in a steep adoption curve.

The verdict is descriptive-only. A sharp, well-localized acceleration in measured AI adoption in AI-exposed sectors did begin with the first BTOS wave after 20 January 2025 — the localization is data-driven and survives composition and placebo-in-space checks — but every test that would pin the bend to DeepSeek-R1 specifically fails: placebo dates match it, most specifications miss it, unexposed sectors show it too, and the o1 contrast sign-flips. Something bent in January 2025; these data cannot say it was R1.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 at seed 0 from the frozen BTOS workbooks (published at <https://www.census.gov/hfp/btos>; not committed to the repository). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the primary estimate against the numbers of record before drawing.

<div class="thebibliography">

9

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bonney, K., Breaux, C., Buffington, C., Dinlersoz, E., Foster, L., Goldschlag, N., Haltiwanger, J., Kroff, Z., and Savage, K. (2024). Tracking firm use of AI in real time: A snapshot from the Business Trends and Outlook Survey. NBER Working Paper No. 32319.

Cameron, A. C., and Miller, D. L. (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 645(8081), 633–638.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

McElheran, K., Li, J. F., Brynjolfsson, E., Kroff, Z., Dinlersoz, E., Foster, L., and Zolas, N. (2024). AI adoption in America: Who, what, and where. *Journal of Economics & Management Strategy*, 33(2), 375–415.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

Yotzov, I., Barrero, J. M., Bloom, N., Bunn, P., Davis, S. J., Foster, K. M., Jalca, A., Meyer, B. H., Mizen, P., Navarrete, M. A., Smietanka, P., Thwaites, G., and Wang, B. Z. (2026). Firm data on AI. NBER Working Paper No. 34836.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.

[^2]: See, e.g., “Nvidia drops \$600B off its market cap amid the rise of DeepSeek,” TechCrunch, 27 January 2025. We do not rely on this financial-markets literature for identification; it motivates the candidate event date only.
