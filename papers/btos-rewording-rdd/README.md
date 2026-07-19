> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/btos-rewording-rdd/) · [PDF in this repo](./main.pdf)

# Introduction

Since September 2023 the Census Bureau’s Business Trends and Outlook Survey (BTOS) has asked a probability sample of US firms whether they used artificial intelligence in the last two weeks; introduce these data, which have become the flagship high-frequency gauge of US business AI adoption — the headline series in the Federal Reserve’s monitoring of the AI economy . On 3 December 2025 the Census Bureau announced that, beginning with survey cycle 2 of sample year 2026, the AI questions were reworded: the core use question now asks about AI “in any of its business functions” rather than “in producing goods or services,” explicitly to reduce respondent under-reporting . Measured adoption moved immediately: the last old-wording wave reads 10.0%, the first new-wording wave 17.3%. Commentary quickly flagged the break — show that how the question is asked drives large gaps between firm AI-adoption estimates, and notes the redesign when reporting the $`\sim`$<!-- -->18% year-end-2025 level — and Census researchers now exploit the expanded business-function detail the redesign introduced . What the commentary does not provide is a formal estimate of the rewording effect with falsification-calibrated uncertainty. That estimate is the number any user of the series needs in order to splice the pre- and post-refresh segments into one panel.

Instrument refreshes breaking measured series is a classic survey-methods problem. The canonical precedent is the 1994 Current Population Survey redesign, whose effects were quantified with a dedicated parallel survey run under both instruments and converted into adjustment factors for splicing the historical series . The BTOS refresh has no parallel overlap sample: the old question stopped, the new one started, and a federal government shutdown sits between them. The splice factor must therefore be estimated from the time series alone — a known-cutoff regression-discontinuity-in-time (RDiT) problem , here in its cleanest form: the cutoff is announced, mechanical, and datable to a single wave, and the “treatment” is the measurement instrument itself, so the usual manipulation and anticipation concerns are moot while the RDiT time-series concerns (serial correlation, trend extrapolation) remain and are addressed directly. We estimate the jump by local linear regression on each side of the cutoff with robust and HAC inference , calibrate it against a 49-point placebo-cutoff grid, bound the adoption-growth confound, and additionally ask the `natex` toolkit to find the discontinuity *blind*, via the LoRD3 local-RD scan of , plus a secondary regression-kink test in the conventions of and .

# Data

The outcome is the national share of firms answering “Yes” to BTOS Question 7, in percentage points, with published survey standard errors. Two vintages are required because current Census downloads carry only the reworded question, with all earlier waves blanked. The old-wording segment (“… did this business use Artificial Intelligence (AI) in producing goods or services?”) comes from a frozen pre-refresh AI-core workbook: 54 biweekly waves, 202319–202520, rising 3.7% to 10.0%. The new-wording segment (“…in any of its business functions?”) comes from a fresh `National.xlsx` download (census.gov, 2026-07-18, 82 kB, verified): 17 waves, 202524–202614, rising 17.3% to 21.7%. The spliced panel has $`n=71`$ waves indexed by $`t`$, the decimal-year reference-period midpoint.

Three timeline facts matter. First, waves 202521–202523 were never collected — the October–November 2025 federal government shutdown — so the design has a built-in donut that is a no-data gap, not censoring: 50 days separate the last pre-cutoff reference midpoint (2025-09-14) from the cutoff. Second, the cutoff is $`t_c = 2025.8384`$ (2025-11-03, the reference-period start of wave 202524, the first wave fielded with the reworded question; published 2025-12-04). No reference midpoints fall in (2025-09-14, 2025-11-09), so any cutoff in that interval yields the identical estimation sample. Third, wave 202524 also begins Sample Year 4 (a scheduled full sample rotation), so the refresh bundles the wording change with a panel rotation — a caveat carried throughout.

# Design and Methods

#### Estimator.

The headline specification fits SE-weighted ($`w=1/\mathrm{se}^2`$) local linear regressions of the AI-use share on $`t - t_c`$ separately on each side of the cutoff within bandwidth $`h=1.0`$ years ($`n_{\text{pre}}=23`$, $`n_{\text{post}}=17`$); the jump $`\tau`$ is the post-side intercept minus the pre-side extrapolation at $`t_c`$. Inference is HC1 per side combined by independence, plus a pooled HAC (Bartlett kernel, 4 and 8 lags) variant for serial correlation. Sensitivity covers bandwidths $`h \in \{0.5, 0.75, 1.0, 2.3\}`$, unweighted fits, and pre-side quadratics.

#### Identification and falsification.

In a measurement RDD the treatment is the instrument refresh, so the only smooth confound is true adoption growth loading into the 50-day extrapolation window; we bound it by the fitted pre-slope and report the estimate under a doubled gap slope. Calibration is design-based: the identical estimator is run at all 49 feasible interior placebo cutoffs, giving a randomization $`p`$-value with floor $`1/50`$. A blind check uses `natex discover` (LoRD3, seed 0, $`k=12`$, $`q=99`$): the scan searches for local discontinuities without being told the cutoff, then honest (split-sample) 2SLS re-estimates the effect at the discovered boundary . The McCrary density test returns NaN on a uniform time grid — inapplicable in RDiT, reported as an outcome, not patched over. A secondary sharp regression-kink test at $`t_c`$ (`natex kink`, right-minus-left slope contrast, policy kink 1, $`h=1.0`$, conventional local-polynomial inference without HAC) asks whether the trend also steepened .

#### Honest-inference notes, stated as run.

The pipeline’s refusals and artifacts are reported: the automated `natex survey` run refused its rdd family (100-row floor, $`n=71`$), iv ($`\geq 80`$), did/sc (needs a panel), and dee ($`\geq 200`$); its kink family auto-assigned the degenerate outcome `run` (a linear transform of $`t`$), so that null ($`p=0.79`$, $`\tau \approx 10^{-13}`$) is meaningless and was superseded by the declared-outcome kink run above. The LoRD3 validation battery passed (covariate placebo Holm $`p=0.24`$); its blind-scan randomization test is underpowered at $`n=71`$ ($`p=0.14`$) and is reported as the null it is. Pre-fit residuals have AR(1) $`=0.836`$ and empirical wave noise $`\sim`$<!-- -->4$`\times`$ the published survey SEs (residual sd $`0.614`$ vs. mean reported se $`0.151`$); the HC1/HAC empirical standard errors absorb this overdispersion, but HAC with $`n_{\text{post}}=17`$ is approximate — the jump is $`\sim`$<!-- -->16 HAC standard errors, so no plausible correction overturns it. A December–January seasonal coefficient is insignificant ($`0.293 \pm 0.236`$).

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) The spliced BTOS national AI-use series with SE-weighted local linear fits within <span class="math inline"><em>h</em> = 1.0</span> yr of the cutoff <span class="math inline"><em>t</em><sub><em>c</em></sub> = 2025.8384</span> (2025-11-03, vertical line); the dashed segment is the 50-day extrapolation across the shutdown gap (shaded); the arrow marks the <span class="math inline">+6.04</span> pp jump. (b) The same estimator at all 49 feasible placebo cutoffs (open circles) versus the true cutoff (filled, with 95% HAC confidence interval): the largest placebo is <span class="math inline">1.30</span> pp, <span class="math inline">4.7×</span> smaller than the true jump (randomization <span class="math inline"><em>p</em> = 0.020</span>).</figcaption>
</figure>

#### Headline jump and robustness.

At the known cutoff the rewording jump is $`\hat\tau = +6.039`$ pp (HC1 se $`0.263`$, $`z=22.9`$; HAC(4) se $`0.371`$, $`z=16.3`$; HAC(8) se $`0.376`$), against an old-wording level of $`10.92`$ pp at the cutoff — a ratio of $`1.553`$. Every variant in Table <a href="#tab:rd" data-reference-type="ref" data-reference="tab:rd">1</a> stays within $`6.0`$–$`7.5`$ pp with $`z>12`$: bandwidths $`0.5`$/$`0.75`$/$`2.3`$ give $`7.148`$/$`6.383`$/$`7.363`$, the unweighted fit gives $`5.956`$, and pre-side quadratics give $`5.787`$–$`7.464`$.

#### Placebo-cutoff calibration.

Across the 49 feasible interior placebo cutoffs the same estimator yields a maximum $`|\tau|`$ of $`1.298`$ pp (mean $`0.515`$); none reaches the true jump in $`|\tau|`$ or $`|z|`$, for a randomization $`p = 0.020`$ — the minimum achievable with 49 placebos (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). The true jump is $`4.7\times`$ the largest placebo.

#### Blind re-localization.

Without being told the cutoff, the LoRD3 scan’s two highest-scoring candidate boundaries sit at $`t=2025.7027`$ and $`t=2025.8562`$ — exactly the last old-wording and first new-wording waves, flanking $`t_c`$. Honest 2SLS at the discovered boundary gives $`\tau = 6.743`$ (se $`0.375`$, 95% CI $`[6.008, 7.478]`$, no weak-instrument flag); the naive full-sample Wald estimate is $`8.06`$ (se $`0.206`$). The scan-level randomization test does not reach significance ($`p=0.14`$), as expected at $`n=71`$; localization, not the scan $`p`$-value, is the evidential contribution.

#### Bounding the adoption confound.

The fitted pre-slope is $`4.89 \pm 0.31`$ pp/yr, which predicts only $`+0.66`$ pp of true-adoption growth across the 50-day gap. Even if the true gap slope were double the fitted slope, the implied jump remains $`\geq 5.376`$ pp. A wording effect of zero would require adoption to grow $`\sim`$<!-- -->9$`\times`$ faster during the shutdown than in the preceding year — and then revert, since the post-cutoff slope ($`6.43`$ pp/yr) is only modestly steeper.

#### Secondary kink.

The sharp RKD at $`t_c`$ estimates a slope change of $`+2.170`$ pp/yr (conventional se $`0.558`$, no HAC): the new-wording series trends steeper than the local old-wording trend. Given conventional inference and the bundling caveats, this is suggestive only; it cautions against assuming the wording effect is a pure level shift far from the cutoff.

#### Splice calibration.

The fitted levels at the cutoff are $`10.92`$ pp (old wording) and $`16.96`$ pp (new wording): an additive splice of $`+6.04`$ pp or a ratio splice of $`1.553`$. Over the observed post period the two disagree increasingly far from the cutoff (a ratio splice scales the steeper post trend); for level-tracking the additive factor is the direct RD estimand, and the ratio is the natural choice if the wording effect scales with the underlying level.

<table id="tab:rd">
<caption>RD estimates of the BTOS rewording jump at <span class="math inline"><em>t</em><sub><em>c</em></sub> = 2025.8384</span>.</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline"><em>z</em></span></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel A: known-cutoff RD jump (pp), SE-weighted local linear</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 1.0</span>, HC1 (headline)</td>
<td style="text-align: right;"><span class="math inline">+6.039</span></td>
<td style="text-align: right;"><span class="math inline">0.263</span></td>
<td style="text-align: right;"><span class="math inline">22.9</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 1.0</span>, HAC Bartlett, 4 lags</td>
<td style="text-align: right;"><span class="math inline">+6.039</span></td>
<td style="text-align: right;"><span class="math inline">0.371</span></td>
<td style="text-align: right;"><span class="math inline">16.3</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 1.0</span>, HAC Bartlett, 8 lags</td>
<td style="text-align: right;"><span class="math inline">+6.039</span></td>
<td style="text-align: right;"><span class="math inline">0.376</span></td>
<td style="text-align: right;"><span class="math inline">16.1</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 0.5</span></td>
<td style="text-align: right;"><span class="math inline">+7.148</span></td>
<td style="text-align: right;"><span class="math inline">0.352</span></td>
<td style="text-align: right;"><span class="math inline">20.3</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 0.75</span></td>
<td style="text-align: right;"><span class="math inline">+6.383</span></td>
<td style="text-align: right;"><span class="math inline">0.275</span></td>
<td style="text-align: right;"><span class="math inline">23.2</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 2.3</span> (full sample)</td>
<td style="text-align: right;"><span class="math inline">+7.363</span></td>
<td style="text-align: right;"><span class="math inline">0.241</span></td>
<td style="text-align: right;"><span class="math inline">30.6</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 1.0</span>, unweighted</td>
<td style="text-align: right;"><span class="math inline">+5.956</span></td>
<td style="text-align: right;"><span class="math inline">0.287</span></td>
<td style="text-align: right;"><span class="math inline">20.8</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 0.75</span>, pre-quadratic</td>
<td style="text-align: right;"><span class="math inline">+7.464</span></td>
<td style="text-align: right;"><span class="math inline">0.481</span></td>
<td style="text-align: right;"><span class="math inline">15.5</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 1.0</span>, pre-quadratic</td>
<td style="text-align: right;"><span class="math inline">+7.051</span></td>
<td style="text-align: right;"><span class="math inline">0.374</span></td>
<td style="text-align: right;"><span class="math inline">18.9</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>h</em> = 2.3</span>, pre-quadratic</td>
<td style="text-align: right;"><span class="math inline">+5.787</span></td>
<td style="text-align: right;"><span class="math inline">0.282</span></td>
<td style="text-align: right;"><span class="math inline">20.5</span></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel B: blind discovery, placebos, and secondary tests</em></td>
</tr>
<tr>
<td style="text-align: left;">LoRD3 honest 2SLS (blind scan)</td>
<td style="text-align: right;"><span class="math inline">+6.743</span></td>
<td style="text-align: right;"><span class="math inline">0.375</span></td>
<td style="text-align: right;"><span class="math inline">18.0</span></td>
</tr>
<tr>
<td style="text-align: left;">LoRD3 naive Wald</td>
<td style="text-align: right;"><span class="math inline">+8.063</span></td>
<td style="text-align: right;"><span class="math inline">0.206</span></td>
<td style="text-align: right;"><span class="math inline">39.2</span></td>
</tr>
<tr>
<td style="text-align: left;">Placebo cutoffs (49): max <span class="math inline">|<em>τ</em>|</span></td>
<td style="text-align: right;"><span class="math inline">1.298</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
</tr>
<tr>
<td style="text-align: left;">Placebo cutoffs (49): mean <span class="math inline">|<em>τ</em>|</span></td>
<td style="text-align: right;"><span class="math inline">0.515</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
</tr>
<tr>
<td style="text-align: left;">Sharp RKD slope change (pp/yr)</td>
<td style="text-align: right;"><span class="math inline">+2.170</span></td>
<td style="text-align: right;"><span class="math inline">0.558</span></td>
<td style="text-align: right;"><span class="math inline">3.9</span></td>
</tr>
</tbody>
</table>

Notes: outcome is the national share answering Yes to BTOS Question 7, percentage points; $`n=23`$ pre / $`17`$ post waves at $`h=1.0`$ yr. HC1 per side combined by independence; HAC is a pooled Bartlett-kernel variant. Placebo randomization $`p=0.020`$ ($`0`$ of $`49`$ placebos $`\geq`$ the true jump in $`|\tau|`$ or $`|z|`$; floor $`1/50`$). LoRD3: `natex discover`, seed 0, $`k=12`$, $`q=99`$; scan randomization $`p=0.14`$ (underpowered at $`n=71`$); McCrary density test NaN (uniform time grid, inapplicable). RKD: conventional local-polynomial inference, no HAC; secondary.

# Caveats and Conclusion

Four caveats bound the claim. First, the donut is a federal-shutdown no-data gap (waves 202521–202523 were never collected), so any true adoption acceleration inside the 50-day extrapolation window loads into $`\hat\tau`$; the pre-slope bound caps this at $`0.66`$ pp under the fitted trend and leaves $`\tau \geq 5.38`$ pp even at double that slope. Second, the questionnaire refresh is bundled with a full sample rotation (Sample Year 4 begins at wave 202524) and follows the shutdown, so “wording effect” strictly means wording $`+`$ panel-refresh $`+`$ shutdown-adjacent response-composition effect; for the splice-factor purpose all three are measurement artifacts, but pure wording attribution is not separable in this design. Third, RDiT caveats apply : pre-fit residual AR(1) is $`0.836`$ and empirical wave noise is $`\sim`$<!-- -->4$`\times`$ the published survey SEs; the HC1/HAC empirical standard errors absorb this, but HAC with 17 post waves is approximate — the jump is $`\sim`$<!-- -->16 HAC standard errors, so no plausible correction overturns it. Fourth, the automated-pipeline artifacts were triaged, not hidden: the survey rdd family refused on its 100-row floor, the auto-assigned degenerate kink outcome was superseded by the declared-outcome run, the McCrary test is inapplicable by construction, and the blind-scan randomization $`p=0.14`$ is an underpowered null.

The verdict is credible: a textbook known-cutoff measurement RDD. The $`+6.04`$ pp jump at the 2025-11-03 refresh is $`4.7\times`$ the largest of 49 placebo jumps, survives every bandwidth, weighting, curvature, and HAC variant within a $`6.0`$–$`7.5`$ pp band, is blindly re-localized by LoRD3 at exactly the flanking waves, and cannot be true adoption. It identifies the instrument-refresh effect and calibrates the splice — additive $`+6.04`$ pp, ratio $`1.553`$ — required to extend any analysis built on the old-wording BTOS AI series past December 2025. Substantively, the estimate implies that roughly a third of the post-refresh measured adoption level is attributable to asking a better question rather than to more firms using AI — a caution for any trend analysis that splices BTOS vintages naively, and a quantification of the under-reporting the Census Bureau redesigned the question to remove .

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 at seed 0 from the frozen BTOS workbooks (published at <https://www.census.gov/hfp/btos>; not committed to the repository). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline estimate, the refitted cutoff levels, and the placebo maximum against the numbers of record before drawing.

<div class="thebibliography">

9

Allen, J. S. (2026). Monitoring AI adoption in the U.S. economy. *FEDS Notes*, Board of Governors of the Federal Reserve System, 3 April 2026.

Bick, A., Blandin, A., Deming, D., Fuchs-Schündeln, N., and Jessen, J. (2026). Measuring AI adoption among firms: How you ask matters. Federal Reserve Bank of St. Louis, *On the Economy*, June 2026.

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bonney, K., Breaux, C., Buffington, C., Dinlersoz, E., Foster, L., Goldschlag, N., Haltiwanger, J., Kroff, Z., and Savage, K. (2024). Tracking firm use of AI in real time: A snapshot from the Business Trends and Outlook Survey. NBER Working Paper No. 32319.

Bonney, K., Breaux, C., Dinlersoz, E., Foster, L., Haltiwanger, J., and Pande, A. (2026). The microstructure of AI diffusion: Evidence from firms, business functions, and worker tasks. Census Bureau CES Working Paper CES-WP-26-25; NBER Working Paper No. 35141.

Calonico, S., Cattaneo, M. D., and Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

U.S. Census Bureau (2025). *Business Trends and Outlook Survey: AI core question updates*. 3 December 2025. <https://www.census.gov/hfp/btos/downloads/AI%20Question%20Wording%20Updates.pdf>

Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

Polivka, A. E., and Miller, S. M. (1998). The CPS after the redesign: Refocusing the economic lens. In *Labor Statistics Measurement Issues*, 249–289. University of Chicago Press.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
