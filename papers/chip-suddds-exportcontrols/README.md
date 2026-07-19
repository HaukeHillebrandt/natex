> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/chip-suddds-exportcontrols/) · [PDF in this repo](./main.pdf)

# Introduction

On 7 October 2022 the US Bureau of Industry and Security (BIS) imposed the first broad export controls on advanced AI accelerators destined for China, a policy contemporaries called a landmark in US–China technology competition . Nvidia responded within weeks with China-market SKUs — the A800 and H800 — engineered just under the control thresholds. On 17 October 2023 BIS closed that loophole: the updated advanced-computing rule adjusted the chip-level performance parameters so that the workaround SKUs themselves became controlled. Nvidia iterated once more with the H20, a Hopper-generation part compliant with the 2023 thresholds — until 9 April 2025, when the US government informed the company that H20 exports to China (and any chip matching its bandwidth profile) would require a license, indefinitely; Nvidia’s securities filing disclosed up to \$5.5 billion of associated charges .

A growing empirical literature evaluates these controls. On the supplier side, show that US firms subject to export controls halt sales to their Chinese customers as intended but largely fail to replace them, losing market value, profitability, and employment. On the evasion side, document underground markets for controlled accelerators and estimate the 2023 smuggling flow at hundreds to low thousands of chips. A companion country-level study in this collection found that the October-2023 round *bent* China’s legal chip-stock growth downward while aggregate compute was substantially bypassed. What that literature leaves untested is the sharpest granularity at which the policy is written: the named SKU. Each control round names specific products; if the rounds bind, product-level shipment panels should carry a break at the declared quarter in exactly the named SKUs — and nowhere else.

This paper runs that test three ways with the `natex` toolkit . Leg 1 asks whether a scan-based difference-in-differences that searches over break dates , *knowing* which chips were treated, recovers the declared policy quarter from the data. Leg 2 removes the treatment labels entirely and asks what the sharpest structure in each panel actually is. Leg 3 estimates effect magnitudes with the placebo-based inference appropriate to a handful of treated clusters . The answer pattern — exact localization everywhere, significance only for October 2023, attribution defeated by product-lifecycle churn — is documented below, refusals included.

# Data

Two Epoch AI datasets are used. *Data on AI Chip Sales* provides quarterly shipment-flow estimates by chip model in thousands of H100-equivalents (kH100e); the extract’s estimates were generated in January 2026 with 2025Q4 the last full-coverage quarter. *Data on AI Chip Components* provides quarterly CoWoS advanced-packaging wafer estimates by chip for 2024Q1–2025Q4. From these, three chip $`\times`$ quarter panels are built (time index $`t=4\,\mathrm{year}+\mathrm{quarter}-1`$, so $`8095=`$ 2023Q4 and $`8101=`$ 2025Q2): (i) *sales Oct-23*, 7 chips $`\times`$ 16 quarters (2022Q1–2025Q4), treated {A800, H800}, declared $`t_0=`$ 2023Q4; (ii) *sales H20*, 20 chips $`\times`$ 8 quarters (2024Q1–2025Q4), treated {H20}, declared $`t_0=`$ 2025Q2; and (iii) *components H20*, 15 chips $`\times`$ 8 quarters of CoWoS wafers, treated {H20, MI308X}, declared $`t_0=`$ 2025Q2.

The outcome is $`\operatorname{asinh}`$ of the flow. Zeros are true zeros (no shipments), not missingness; chips born after $`t_0-1`$ are excluded at build time so that treatment cannot be mechanically confounded with birth; and the ragged 2026 tail is dropped at the 2025Q4 snapshot. Because the treatment effect here operates on the extensive margin — banned SKUs’ shipments go to approximately zero — asinh magnitudes are unit-dependent and should not be read as percentage effects ; the qualitative reading (shipments driven to zero) is the transform-robust statement. One measurement caveat is carried through the paper: Huawei Ascend and Cambricon Siyuan flows in the source data are annual allocations smeared uniformly across quarters, which makes those series smooth by construction.

# Design and Methods

#### Leg 1: known-treatment scans (date localization).

For each panel, the `natex` SuDDDS scan maximizes a likelihood-ratio statistic for a declared-treatment DiD over candidate break quarters $`t_0`$ and window widths $`W\in\{2,4,8\}`$. Because the unit column is the chip itself, no subset search is available to the known-treatment scan: it localizes over $`(t_0,W)`$ only, so *date localization is the contest* — does the maximum-LLR break land on the declared quarter? The primary model is Bernoulli with within-cluster calibration (wcc); a normal-model cross-check is calibrated under a dependence-preserving AR(1)-by-unit null. Permutation calibration uses $`Q=99`$ draws at seed 0, so the smallest attainable scan $`p`$ is $`0.01`$.

#### Leg 2: unsupervised discovery rematch.

The same scan machinery is then run with the treatment column deleted: $`\theta`$ is the observed flow, a categorical chip-name dimension is admitted to the subset search (model normal, method single_delta, restarts 8, panel-null $`Q=99`$, seed 0), and the scan reports whatever subset $`\times`$ window is sharpest. Discovery never reads treatment labels; the question is whether the policy set is the panel’s sharpest structure or merely sits inside something larger.

#### Leg 3: effects.

A manual two-way fixed-effects (TWFE) DiD with chip and quarter fixed effects gives $`\hat\tau`$ in asinh units with CR1 standard errors clustered by chip. With at most 7 clusters, CR1 $`t`$-statistics are known to over-reject — and demonstrably do here — so the *inference of record* is an exact placebo-in-space test in the spirit of : every same-size subset of control chips is pseudo-treated (true treated units dropped) and $`p`$ is the rank of $`|\hat\tau|`$ in that exact distribution. The preferred specification is lifecycle-matched: donor chips alive (nonzero) in all 8 quarters, removing births and deaths from the donor pool. Robustness: dropping 2025Q1 (a visible pull-forward spike of 54.2k H100e before the April license), restricting to non-Chinese controls (substitution-contamination check), a placebo-in-time at pseudo-$`t_0=`$ 2024Q4 on pre-period data only, and an event study relative to 2025Q1.

#### Honest refusals, stated as run.

`natex`’s built-in dd/synthetic-control/GESS effect estimators refused on all three panels (0 usable placebo units; $`\geq 5`$ required) — hence the manual TWFE leg. The scan’s anticipation test refused on all known-treatment runs (no usable pre-quarters at these window widths; reported as not-passed with $`p=\mathrm{None}`$, a refusal rather than a failure). And the lifecycle-matched Oct-23 effect leg refused outright: 0 of 5 control chips are alive in all 16 quarters, so no matched donor pool exists. Refusals are reported as refusals, not patched over.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Oct-2023 round: <span class="math inline">asinh </span> shipment flows for the banned SKUs (A800, H800) against the five incumbent control chips; the scan puts the break exactly at the declared quarter (dashed line, 2023Q4 boundary) with <span class="math inline"><em>p</em> = 0.030</span>, and treated shipments hit literal zero within one quarter. (b) H20 round: the H20 against 6 lifecycle-matched controls (solid) and 13 unmatched chips entering or exiting the market (dotted) — the churn that defeats attribution; the break again lands exactly on the declared quarter but at <span class="math inline"><em>p</em> = 0.44</span>. (c) Event study, H20 minus matched controls relative to 2025Q1: flat pre-trend, monotone post-collapse to <span class="math inline">−4.07</span> asinh units by 2025Q4; the shaded band is the placebo-in-space envelope (max <span class="math inline">|<em>τ</em>|</span> over the 6 matched controls), which the pooled estimate exceeds only at the <span class="math inline"><em>p</em> = 1/7</span> floor.</figcaption>
</figure>

#### Localization: six for six, one significant.

Every known-treatment scan puts the maximum-LLR break at the declared quarter exactly (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A). For the Oct-2023 round (7 chips $`\times`$ 16 quarters, treated A800+H800) the Bernoulli scan discovers $`t_0=8095=`$ 2023Q4 with $`W=4`$ and LLR $`=4.896`$, scan $`p=0.030`$ ($`Q=99`$); the normal/ar1_unit cross-check finds the same $`t_0`$ and $`W`$ with LLR $`=4.085`$, $`p=0.14`$. The banned SKUs’ shipments go to literal zero within one quarter (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>a). For the H20 round the localization is equally exact but never significant: sales panel $`p=0.44`$ (Bernoulli) and $`0.60`$ (normal); components panel (treated H20+MI308X) $`p=0.13`$ and $`0.23`$. The composition check passes ($`p=1.0`$) in all six scans, and role-assignment sanity passes throughout: the scans are dating the declared treatment, not rediscovering an artifact.

#### Discovery: the generation, never the policy set.

With treatment labels removed (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel B), the Oct-23 panel’s sharpest structure is {A800, H100/H200, H800, TPU v4, TPU v5e} at *2022Q4* (LLR $`=16.16`$, $`p=0.01`$ floor) — the Hopper production ramp and the birth of the A800, itself the workaround SKU created by the October-*2022* round. The ban quarter 2023Q4 does not appear in the top five discoveries; the closest is {A100, A800, H800} at 2024Q1 (rank 3) — the banned SKUs plus the A100’s end-of-life. On the H20 sales panel, discovery puts the sharpest break at 2025Q2 *exactly* (LLR $`=17.32`$, $`p=0.05`$), but the discovered subset is {H100/H200, H20, MI300A, TPU v5e, TPU v5p, Trainium1} — the H20 merged into the entire Hopper-generation wind-down, with the complementary Blackwell-generation upshift as the runner-up. The components panel repeats the pattern one quarter early (7-chip old-generation set at 2025Q1, LLR $`=16.41`$, $`p=0.02`$). Every autonomously discovered subset is a chip *generation*, never the policy set: at product granularity, the H20 ban and the Hopper-to-Blackwell transition are the same event time.

#### Effects: large, sign-correct, unattributable.

The lifecycle-matched TWFE estimate for the H20 (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel C) is $`\hat\tau=-2.756`$ asinh units (CR1 se $`0.652`$, $`t=-4.23`$) with 0 of 6 matched placebos exceeding — but 6 controls cap the exact test at its $`p=1/7=0.143`$ floor. The event study is clean in shape: pre-period coefficients $`+0.19`$, $`+0.37`$, $`+0.29`$, $`+0.01`$, $`0`$; post-period $`-0.80`$, $`-2.88`$, $`-4.07`$ (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>c). The naive (unmatched) legs reproduce the lifecycle failure and calibrate how oversized CR1 is on these panels: the naive H20 spec has $`t=-4.09`$ yet 7 of 19 placebo chips produce larger pseudo-effects ($`p=0.400`$); naive Oct-23 gives $`p=0.727`$; naive components $`p=0.203`$. Robustness on the matched leg: dropping the 2025Q1 anticipation spike leaves $`\hat\tau=-2.799`$ ($`t=-3.73`$) but costs one placebo exceedance ($`p=0.286`$); restricting to the 3 non-Chinese controls gives $`\hat\tau=-2.018`$ with placebo $`p=0.750`$ — the contamination check fails its placebo test, with the H100/H200’s own Blackwell wind-down ($`-2.12`$) and Trainium2 ($`+2.88`$) both comparable to the estimate. The placebo-in-time is clean ($`\hat\tau=-0.280`$, $`t=-0.67`$).

<table id="tab:main">
<caption>Headline numbers of record, all three legs. Panel A: known-treatment SuDDDS scans; the declared quarter is 2023Q4 for the Oct-23 panel and 2025Q2 for both H20 panels, and <span class="math inline"><em>t̂</em><sub>0</sub></span> is the scan’s maximum-LLR break quarter. Panel B: unsupervised discovery (no treatment column; <span class="math inline"><em>θ</em>=</span> the flow). Panel C: TWFE effects in asinh units; the exact placebo-in-space <span class="math inline"><em>p</em></span> is the inference of record.</caption>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: known-treatment scans (date localization)</em></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: right;"><span class="math inline"><em>t̂</em><sub>0</sub></span></td>
<td style="text-align: right;"><span class="math inline"><em>W</em></span></td>
<td style="text-align: right;">LLR</td>
<td style="text-align: right;">scan <span class="math inline"><em>p</em></span></td>
</tr>
<tr>
<td style="text-align: left;">sales Oct-23 (<span class="math inline">7 × 16</span>; A800+H800), Bernoulli/wcc</td>
<td style="text-align: right;">2023Q4 exact</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">4.896</span></td>
<td style="text-align: right;"><span class="math inline">0.030</span></td>
</tr>
<tr>
<td style="text-align: left;">normal/ar1_unit cross-check</td>
<td style="text-align: right;">2023Q4 exact</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">4.085</span></td>
<td style="text-align: right;"><span class="math inline">0.14</span></td>
</tr>
<tr>
<td style="text-align: left;">sales H20 (<span class="math inline">20 × 8</span>; H20), Bernoulli/wcc</td>
<td style="text-align: right;">2025Q2 exact</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">1.532</span></td>
<td style="text-align: right;"><span class="math inline">0.44</span></td>
</tr>
<tr>
<td style="text-align: left;">normal/ar1_unit cross-check</td>
<td style="text-align: right;">2025Q2 exact</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;"><span class="math inline">0.914</span></td>
<td style="text-align: right;"><span class="math inline">0.60</span></td>
</tr>
<tr>
<td style="text-align: left;">components H20 (<span class="math inline">15 × 8</span>; H20+MI308X), Bernoulli/wcc</td>
<td style="text-align: right;">2025Q2 exact</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">2.331</span></td>
<td style="text-align: right;"><span class="math inline">0.13</span></td>
</tr>
<tr>
<td style="text-align: left;">normal/ar1_unit cross-check</td>
<td style="text-align: right;">2025Q2 exact</td>
<td style="text-align: right;">2</td>
<td style="text-align: right;"><span class="math inline">1.949</span></td>
<td style="text-align: right;"><span class="math inline">0.23</span></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: unsupervised discovery (top subset; every one a chip generation)</em></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: right;"><span class="math inline"><em>t̂</em><sub>0</sub></span></td>
<td style="text-align: right;"><span class="math inline"><em>W</em></span></td>
<td style="text-align: right;">LLR</td>
<td style="text-align: right;"><span class="math inline"><em>p</em></span></td>
</tr>
<tr>
<td style="text-align: left;">sales Oct-23: {A800, H100/H200, H800, TPU v4, TPU v5e}</td>
<td style="text-align: right;">2022Q4</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">16.16</span></td>
<td style="text-align: right;"><span class="math inline">0.01</span> (floor)</td>
</tr>
<tr>
<td style="text-align: left;">sales H20: {H100/H200, H20, MI300A, TPU v5e/v5p, Trainium1}</td>
<td style="text-align: right;">2025Q2 exact</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">17.32</span></td>
<td style="text-align: right;"><span class="math inline">0.05</span></td>
</tr>
<tr>
<td style="text-align: left;">components H20: 7-chip old-generation set (incl. H20, MI308X)</td>
<td style="text-align: right;">2025Q1</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;"><span class="math inline">16.41</span></td>
<td style="text-align: right;"><span class="math inline">0.02</span></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: TWFE effects (asinh units; exceedances <span class="math inline">=</span> placebos <span class="math inline"> ≥ |<em>τ̂</em>|</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: right;"><span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: right;">CR1 se</td>
<td style="text-align: right;"><span class="math inline"><em>t</em></span></td>
<td style="text-align: right;">exceed., <span class="math inline"><em>p</em></span></td>
</tr>
<tr>
<td style="text-align: left;">matched sales H20 (6 controls alive all 8 q)</td>
<td style="text-align: right;"><span class="math inline">−2.756</span></td>
<td style="text-align: right;"><span class="math inline">0.652</span></td>
<td style="text-align: right;"><span class="math inline">−4.23</span></td>
<td style="text-align: right;">0/6, <span class="math inline">0.143</span> (floor)</td>
</tr>
<tr>
<td style="text-align: left;">drop 2025Q1 (anticipation spike)</td>
<td style="text-align: right;"><span class="math inline">−2.799</span></td>
<td style="text-align: right;"><span class="math inline">0.751</span></td>
<td style="text-align: right;"><span class="math inline">−3.73</span></td>
<td style="text-align: right;">1/6, <span class="math inline">0.286</span></td>
</tr>
<tr>
<td style="text-align: left;">non-Chinese controls only (3)</td>
<td style="text-align: right;"><span class="math inline">−2.018</span></td>
<td style="text-align: right;"><span class="math inline">0.939</span></td>
<td style="text-align: right;"><span class="math inline">−2.15</span></td>
<td style="text-align: right;">2/3, <span class="math inline">0.750</span></td>
</tr>
<tr>
<td style="text-align: left;">naive sales Oct-23 (all 5 controls)</td>
<td style="text-align: right;"><span class="math inline">−1.070</span></td>
<td style="text-align: right;"><span class="math inline">1.160</span></td>
<td style="text-align: right;"><span class="math inline">−0.92</span></td>
<td style="text-align: right;">7/10, <span class="math inline">0.727</span></td>
</tr>
<tr>
<td style="text-align: left;">naive sales H20 (all 19 controls)</td>
<td style="text-align: right;"><span class="math inline">−2.347</span></td>
<td style="text-align: right;"><span class="math inline">0.574</span></td>
<td style="text-align: right;"><span class="math inline">−4.09</span></td>
<td style="text-align: right;">7/19, <span class="math inline">0.400</span></td>
</tr>
<tr>
<td style="text-align: left;">naive components H20 (all 13 controls)</td>
<td style="text-align: right;"><span class="math inline">−2.059</span></td>
<td style="text-align: right;"><span class="math inline">0.876</span></td>
<td style="text-align: right;"><span class="math inline">−2.35</span></td>
<td style="text-align: right;">15/78, <span class="math inline">0.203</span></td>
</tr>
<tr>
<td style="text-align: left;">placebo-in-time (pseudo-<span class="math inline"><em>t</em><sub>0</sub></span> 2024Q4, pre-only)</td>
<td style="text-align: right;"><span class="math inline">−0.280</span></td>
<td style="text-align: right;"><span class="math inline">0.420</span></td>
<td style="text-align: right;"><span class="math inline">−0.67</span></td>
<td style="text-align: right;">clean</td>
</tr>
<tr>
<td style="text-align: left;">matched Oct-23 analog</td>
<td colspan="4" style="text-align: left;">refused: 0 of 5 control chips alive all 16 quarters</td>
</tr>
</tbody>
</table>

Notes: all runs `natex` v0.2.0, seed 0. Scan $`p`$-values from permutation calibration with $`Q=99`$ (floor $`0.01`$); Bernoulli scans use within-cluster calibration (wcc), normal scans a dependence-preserving AR(1)-by-unit null. Composition checks pass ($`p=1.0`$) in all six panel-A scans; the anticipation test refused (no usable pre-quarters). CR1 clusters by chip; with $`G\leq 7`$ the exact placebo-in-space $`p`$ is the inference of record, and the naive H20 row ($`t=-4.09`$, placebo $`p=0.400`$) calibrates CR1’s oversizing on these panels. Matched controls: Ascend 910B, Ascend 910C, H100/H200, MI300X, Siyuan 590, Trainium2 (placebo $`\hat\tau`$: $`-1.03`$, $`+2.06`$, $`-2.58`$, $`-1.50`$, $`+1.63`$, $`+1.42`$).

# Caveats and Conclusion

Four caveats bound the claims. First, *product-lifecycle churn is a competing sharp treatment*: unsupervised discovery finds only chip-generation subsets at the policy quarters — the Hopper wind-down with the H20 merged into it at 2025Q2, the Hopper ramp with the A800’s birth at 2022Q4 — so at product granularity the H20 ban and the Blackwell transition are colinear in event time and cannot be separated inside this dataset. Second, *control contamination*: 3 of the 6 lifecycle-matched controls are Chinese accelerators (Ascend 910B/910C, Siyuan 590) that plausibly *gained* demand from the ban — a SUTVA violation biasing $`\hat\tau`$ negative (their placebo $`\hat\tau`$’s are $`+2.06`$, $`+1.63`$, $`-1.03`$) — while the no-China robustness leg (3 controls) gives $`\hat\tau=-2.02`$ with placebo $`p=0.75`$: the matched leg’s 0/6 leans on contaminated donors. Third, *measurement*: Huawei and Cambricon flows are annual allocations smeared over quarters — smooth by construction, understating placebo dispersion — and CR1 $`t`$-statistics are demonstrably oversized here (the naive H20 leg pairs $`t=-4.09`$ with placebo $`p=0.400`$), which is why the exact placebo-in-space $`p`$ is the inference of record. Fourth, *resolution floors*: 6 matched controls cap the placebo test at $`p=1/7=0.143`$ and the scan permutation floor is $`0.01`$ ($`Q=99`$); no $`p`$-value below its floor is claimed. Relatedly, all treated units share one policy date per round, so scan $`p`$-values come from the fitted-null panel calibration rather than cross-unit timing variation — the Bernoulli $`p=0.030`$ for Oct-23 is the defensible headline, with the normal cross-check ($`p=0.14`$) reported beside it.

The verdict is deliberately asymmetric. Date localization at product granularity *works*: all six known-treatment scans land on the declared quarter exactly, the October-2023 round clears its calibrated null ($`p=0.030`$), and the banned SKUs’ shipments hit literal zero within one quarter — a policy written against named products is visible in product-level data on the dot. Attribution of effect *size* at this granularity does not work, and the pipeline says so rather than manufacturing a significant number: the H20 collapse is large ($`\hat\tau\approx -2.8`$ asinh units) and correctly signed, but placebo chips generate ban-sized effects from ordinary lifecycle churn, the clean-donor robustness fails its own placebo test, and the matched Oct-23 leg refuses for want of donors. For an automated natural-experiment pipeline, the refusals and the correctly identified confound are the point: the machinery distinguished what these panels can establish — *when* — from what they cannot — *how much*.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 at seed 0 from frozen extracts of the Epoch AI datasets (not committed to the repository). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline scan and effect estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Allen, G. C. (2022). Choking off China’s access to the future of AI. Center for Strategic and International Studies (CSIS), October 2022. <https://www.csis.org/analysis/choking-chinas-access-future-ai>

Bureau of Industry and Security (2023). Implementation of additional export controls: Certain advanced computing items; supercomputer and semiconductor end use; updates and corrections. Interim final rule, 88 FR 73458, 25 October 2023.

Cameron, A. C., and Miller, D. L. (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Chen, J., and Roth, J. (2024). Logs with zeros? Some problems and solutions. *Quarterly Journal of Economics*, 139(2), 891–936.

Conley, T. G., and Taber, C. R. (2011). Inference with “difference in differences” with a small number of policy changes. *Review of Economics and Statistics*, 93(1), 113–125.

Crosignani, M., Han, L., Macchiavelli, M., and Silva, A. F. (2025). Securing technological leadership? The cost of export controls on firms. Federal Reserve Bank of New York Staff Report No. 1096 (first circulated 2023 as “Geopolitical Risk and Decoupling: Evidence from U.S. Export Controls”).

Epoch AI (2026). Data on AI chip components. <https://epoch.ai/data/ai-chip-components> (extract of 2026, quarters 2024Q1–2025Q4).

Epoch AI (2026). Data on AI chip sales. <https://epoch.ai/data/ai-chip-sales> (extract of 2026; estimates generated January 2026).

Fist, T., and Grunewald, E. (2023). *Preventing AI Chip Smuggling to China: A Working Paper*. Center for a New American Security (CNAS), October 2023.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

Hillebrandt, H. (2026). Bent, bypassed, unbowed: US AI-chip export controls and China’s compute. `natex` paper collection. <https://haukehillebrandt.github.io/natex/export-controls-three-leg/>

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

NVIDIA Corporation (2025). Current report (Form 8-K), 9 April 2025: H20 export-license requirement and up to \$5.5 billion of associated charges. US Securities and Exchange Commission.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
