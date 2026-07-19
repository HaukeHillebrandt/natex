> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/eci-fresh-kink-o1/) · [PDF](https://haukehillebrandt.github.io/natex/eci-fresh-kink-o1.pdf) · [PDF in this repo](./main.pdf)

# Introduction

OpenAI released o1-preview on 12 September 2024, the first widely deployed “reasoning model” trained with large-scale reinforcement learning to spend inference-time compute on a chain of thought . The release is a natural candidate for a regime change in AI capabilities: contemporaneous research argued that scaling test-time compute can beat scaling parameters , and the reasoning-model recipe was adopted across frontier labs within months. If the recipe changed the *rate* of capability progress, an aggregate capability index should show a slope change — a kink — in calendar time at or shortly after the release.

Measuring that requires a scale on which models years apart are comparable. The Epoch Capabilities Index (ECI) stitches dozens of benchmarks onto a single latent-ability scale via an item-response-theory model, precisely to support longitudinal comparisons of this kind . This paper asks a narrow question: does the ECI trend kink at o1?

The estimand — a change in trend slope at a known date — is the regression kink design (RKD) of , applied here with calendar time as the running variable; formalize the closely related difference-in-kinks design that the same toolkit implements. Two strands of methodological literature discipline the exercise. First, regression discontinuity (and kink) designs in *time* lack the randomization logic of cross-sectional designs: units near the cutoff are not quasi-randomly assigned, and serial dependence and smooth trends generate spurious breaks . Second, kink estimates on smooth series are known to produce spuriously significant slope changes, which is why placebo distributions at non-event dates are the appropriate yardstick . Both concerns are operationalized below through a mandatory placebo-cutoff battery and curvature checks, estimated with the `natex` toolkit .

This note is a robustness upgrade of a prior null: an earlier analysis pass on a 2025 vintage of the index found no kink at o1. The fresh vintage adds twelve months of post-cutoff data and per-model confidence intervals, which enable a precision-weighted re-estimate — the natural next test, since noisy early-2023 models could in principle mask a true kink in the unweighted fit.

# Data

The outcome is the fresh vintage of Epoch AI’s `eci_scores.csv` (fetched July 2026): 211 aggregated models with release dates from 2023-02-24 to 2026-07-09 and no missing score or date, spanning ECI $`\approx55`$ to $`161.77`$. The vintage adds roughly twelve months of data relative to the prior analysis pass. Each model carries a 95% confidence interval; the per-model precision weight is $`w_i = 1/\sigma_i^2`$ with $`\sigma_i = (\mathrm{ci}_{\mathrm{high},i} - \mathrm{ci}_{\mathrm{low},i})/3.92`$. Two models (Claude 3.5 Sonnet and GPT-5, the scale’s two anchor models) have no published CI; their $`\sigma_i`$ is imputed at the sample median ($`2.005`$) for the weighted runs only.

A version-level Confidence flag is attached by a two-pass name join against Epoch’s per-version index file; 189 of 211 models match (the 22 unmatched are treated as not-flagged), and 26 models (12.3%) are flagged Unverified/Speculative — including the current \#1, GPT-5.6 Sol (ECI $`161.77`$), and GPT-5. A high-confidence subset excludes these 26 models ($`n=185`$).

Two series are analyzed. The *all-models* series uses all 211 releases. The *clean frontier-records* series keeps strict running-maximum records of the score (computed over non-missing scores only, in date order, with a stable same-day tie-break): 24 records, 7 pre-cutoff and 17 post-cutoff — and only 4 pre-cutoff points inside bandwidth 540 days, 3 after a 45-day donut, which already flags it as a marginal design.

# Design and Methods

#### Sharp RKD in calendar time.

The running variable is days since the o1-preview release (cutoff $`= 0`$ at 2024-09-12). The estimand is the change in the slope of ECI in points/day at the cutoff, estimated by local linear regression on each side within the bandwidth, triangular kernel, HC1 standard errors (`natex` v0.2.0 , `natex kink` with a unit policy-kink change; ). The primary specification uses bandwidth 540 days, donut 0 ($`n=179`$; 85 left, 94 right). A specification grid varies bandwidth $`\in\{365,540,730\}`$ days $`\times`$ kernel $`\in\{\text{triangular},\text{uniform}\}`$ $`\times`$ donut $`\in\{0,30,45\}`$ days (18 cells per series–weighting pair). The CI-weighted estimator is a local WLS that mirrors `natex` exactly (same selection, kernel, block design, HC1) with weights kernel$`\,\times\,1/\sigma_i^2`$; with unit weights it reproduces `natex regression_kink` in all 68 evaluable grid cells (tolerance $`10^{-8}`$ on points, $`10^{-6}`$ on SEs) and matches the NaN reasons in the rest, and the CLI headline cells match the grid to all printed digits. The design is deterministic; seed 0 is declared and no stochastic step exists.

#### Honest-inference caveats, stated as run.

This is a calendar-time RKD: model releases are not randomized around the cutoff, so the cross-sectional RD identification logic does not apply and smooth trends can masquerade as kinks . HC1 treats observations as independent, ignoring same-day multi-releases and lab-level clustering. The placebo-cutoff grid — the same specification re-estimated at $`\pm90`$, $`\pm180`$, $`\pm270`$, $`\pm360`$ days, in the spirit of — is therefore the operative guard, and below it is what both establishes power for the null and disqualifies the weighted “significant” cell. A local-quadratic (degree-2) refit checks whether any kink survives allowing smooth curvature; a release-density kink test (binned 30-day release counts) checks for composition churn at the cutoff.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) The 211 fresh-vintage ECI scores against days since the o1-preview release (vertical line), with per-side local-linear fits (bandwidth 540 days, triangular kernel): unweighted (blue) and CI-weighted (orange). The unweighted slopes are statistically indistinguishable across the cutoff; the CI-weighted fit bends down. (b) Kink estimates with 95% HC1 confidence intervals at the o1 cutoff and at eight placebo cutoffs, both weightings (points offset horizontally for legibility); filled markers are nominally significant at the 5% level. The unweighted design rejects at 4/8 placebo cutoffs while reading zero at o1; the CI-weighted negative kink at o1 is matched by same-sign, larger placebos at <span class="math inline">+90</span> and <span class="math inline">+180</span> days.</figcaption>
</figure>

#### The unweighted null, now with more power.

The all-models headline kink (Table <a href="#tab:kink" data-reference-type="ref" data-reference="tab:kink">1</a>, first row) is $`+0.00653`$ ECI points/day (se $`0.01308`$, $`t=0.50`$, $`p=0.618`$, 95% CI $`[-0.0191, +0.0322]`$): the trend runs $`0.0611`$ points/day before o1 and $`0.0677`$ after ($`\approx22.3`$ vs. $`\approx24.7`$ points/yr). Of the 18 grid cells, 17 are null; the single nominal rejection (bandwidth 365, triangular, donut 30: $`-0.0547`$, se $`0.0269`$, $`t=-2.03`$, $`p=0.042`$) is the grid’s smallest cell and negative. The placebo grid shows the null is not for lack of power: at bandwidth 540 the same specification rejects at 4 of 8 non-event cutoffs ($`-180`$ days: $`+0.0401`$, $`p=0.023`$; $`+180`$: $`-0.0323`$, $`p=7.1\times10^{-4}`$; $`+270`$: $`-0.0288`$, $`p=1.0\times10^{-4}`$; $`+360`$: $`-0.0247`$, $`p=0.0034`$) while the true cutoff is null in all cells — the instrument moves elsewhere and reads zero at o1. The prior-vintage null replicates on twelve more months of data.

#### The CI-weighted “kink” is a curvature artifact.

Precision weighting flips the headline to a nominally significant *negative* kink: $`-0.01903`$ (se $`0.00788`$, $`t=-2.41`$, $`p=0.0158`$), significant in 18 of 18 grid cells ($`|t|`$ from $`2.12`$ to $`3.32`$), with slopes $`0.0522`$ pre vs. $`0.0331`$ post ($`\approx19.0`$ vs. $`\approx12.1`$ points/yr). Two facts disqualify an o1-localized reading. First, the placebo grid rejects with the *same sign and larger or comparable magnitude* away from the release: $`-0.0399`$ ($`p=0.0064`$) at $`+90`$ days, $`-0.0297`$ ($`p=5.5\times10^{-6}`$) at $`+180`$, $`-0.0154`$ ($`p=0.021`$) at $`+270`$ (empirical size 3/8 at $`\alpha=0.05`$). Second, a local-quadratic fit absorbs it: degree-2 kinks are $`-0.0392`$ ($`t=-1.48`$, $`p=0.140`$) at bandwidth 540 and $`-0.0381`$ ($`t=-1.74`$, $`p=0.081`$) at bandwidth 730. The precision-weighted index decelerates smoothly through 2025 — a concavity, not a kink at o1.

#### The frontier-records series is still not a design.

With only 4 left-cell points at bandwidths $`\leq540`$ (3 after donut 45), the records series produces a nominal positive kink at the headline specification ($`+0.01313`$, se $`0.00576`$, $`t=2.28`$, $`p=0.023`$) that no perturbation survives: a 30-day donut kills it ($`t=0.07`$, $`p=0.943`$), a 45-day donut flips it to significantly *negative* ($`-0.02817`$, $`t=-2.52`$, $`p=0.012`$), bandwidth 730 flips the sign ($`-0.00429`$, $`t=-0.27`$), and the placebo grid rejects 5/8 — with the $`-360`$/$`-270`$ cells exploding to $`-1.14`$ ($`p=3\times10^{-4}`$) on $`n=11`$. The CI-weighted records headline is null ($`+0.00794`$, $`t=1.31`$, $`p=0.191`$). The fresh vintage does not clear the left-cell bar: the series remains non-estimable, exactly as in the prior pass.

#### Robustness and diagnostics.

Excluding the 26 Unverified/Speculative models ($`n=185`$) changes nothing: unweighted $`+0.01248`$ ($`t=0.88`$, $`p=0.379`$, null), CI-weighted $`-0.02109`$ ($`t=-2.32`$, $`p=0.020`$) with the same placebo failure (weighted empirical size 4/8, unweighted 6/8). The release-density check finds no composition-churn kink at the cutoff: binned 30-day release counts give $`-0.00410`$ (se $`0.00967`$, $`t=-0.42`$, $`p=0.672`$), an improvement on the prior vintage’s marginal $`t=1.82`$.

<table id="tab:kink">
<caption>Sharp RKD in time at o1-preview (2024-09-12), ECI points/day. Headline specification: bandwidth 540 days, triangular kernel, donut 0, HC1.</caption>
<thead>
<tr>
<th style="text-align: left;">Series</th>
<th style="text-align: left;">Weighting</th>
<th style="text-align: right;">Kink</th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline"><em>t</em></span></th>
<th style="text-align: right;"><span class="math inline"><em>p</em></span></th>
<th style="text-align: right;"><span class="math inline"><em>n</em></span></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="7" style="text-align: left;"><em>Panel A: headline cells</em></td>
</tr>
<tr>
<td style="text-align: left;">All models (211)</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">+0.00653</span></td>
<td style="text-align: right;"><span class="math inline">0.01308</span></td>
<td style="text-align: right;"><span class="math inline">0.50</span></td>
<td style="text-align: right;"><span class="math inline">0.618</span></td>
<td style="text-align: right;">179</td>
</tr>
<tr>
<td style="text-align: left;">All models</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">−0.01903</span></td>
<td style="text-align: right;"><span class="math inline">0.00788</span></td>
<td style="text-align: right;"><span class="math inline">−2.41</span></td>
<td style="text-align: right;"><span class="math inline">0.016</span></td>
<td style="text-align: right;">179</td>
</tr>
<tr>
<td style="text-align: left;">Frontier records (24)</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">+0.01313</span></td>
<td style="text-align: right;"><span class="math inline">0.00576</span></td>
<td style="text-align: right;"><span class="math inline">2.28</span></td>
<td style="text-align: right;"><span class="math inline">0.023</span></td>
<td style="text-align: right;">18</td>
</tr>
<tr>
<td style="text-align: left;">Frontier records</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">+0.00794</span></td>
<td style="text-align: right;"><span class="math inline">0.00608</span></td>
<td style="text-align: right;"><span class="math inline">1.31</span></td>
<td style="text-align: right;"><span class="math inline">0.191</span></td>
<td style="text-align: right;">18</td>
</tr>
<tr>
<td style="text-align: left;">High-confidence (185)</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">+0.01248</span></td>
<td style="text-align: right;"><span class="math inline">0.01420</span></td>
<td style="text-align: right;"><span class="math inline">0.88</span></td>
<td style="text-align: right;"><span class="math inline">0.379</span></td>
<td style="text-align: right;">162</td>
</tr>
<tr>
<td style="text-align: left;">High-confidence</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">−0.02109</span></td>
<td style="text-align: right;"><span class="math inline">0.00908</span></td>
<td style="text-align: right;"><span class="math inline">−2.32</span></td>
<td style="text-align: right;"><span class="math inline">0.020</span></td>
<td style="text-align: right;">162</td>
</tr>
<tr>
<td style="text-align: left;">Release density (30-d bins)</td>
<td style="text-align: left;">—</td>
<td style="text-align: right;"><span class="math inline">−0.00410</span></td>
<td style="text-align: right;"><span class="math inline">0.00967</span></td>
<td style="text-align: right;"><span class="math inline">−0.42</span></td>
<td style="text-align: right;"><span class="math inline">0.672</span></td>
<td style="text-align: right;">36</td>
</tr>
<tr>
<td colspan="7" style="text-align: left;"><em>Panel B: triage of the nominally significant cells</em></td>
</tr>
<tr>
<td style="text-align: left;">All models, CI-w., placebo <span class="math inline">+90</span> d</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">−0.03986</span></td>
<td style="text-align: right;"><span class="math inline">0.01462</span></td>
<td style="text-align: right;"><span class="math inline">−2.73</span></td>
<td style="text-align: right;"><span class="math inline">0.0064</span></td>
<td style="text-align: right;">181</td>
</tr>
<tr>
<td style="text-align: left;">All models, CI-w., placebo <span class="math inline">+180</span> d</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">−0.02966</span></td>
<td style="text-align: right;"><span class="math inline">0.00653</span></td>
<td style="text-align: right;"><span class="math inline">−4.54</span></td>
<td style="text-align: right;"><span class="math inline">5.5 × 10<sup>−6</sup></span></td>
<td style="text-align: right;">174</td>
</tr>
<tr>
<td style="text-align: left;">All models, CI-w., degree 2, bw 540</td>
<td style="text-align: left;">CI-weighted</td>
<td style="text-align: right;"><span class="math inline">−0.03922</span></td>
<td style="text-align: right;"><span class="math inline">0.02659</span></td>
<td style="text-align: right;"><span class="math inline">−1.48</span></td>
<td style="text-align: right;"><span class="math inline">0.140</span></td>
<td style="text-align: right;">179</td>
</tr>
<tr>
<td style="text-align: left;">Frontier records, donut 30</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">+0.00048</span></td>
<td style="text-align: right;"><span class="math inline">0.00667</span></td>
<td style="text-align: right;"><span class="math inline">0.07</span></td>
<td style="text-align: right;"><span class="math inline">0.943</span></td>
<td style="text-align: right;">16</td>
</tr>
<tr>
<td style="text-align: left;">Frontier records, donut 45</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">−0.02817</span></td>
<td style="text-align: right;"><span class="math inline">0.01118</span></td>
<td style="text-align: right;"><span class="math inline">−2.52</span></td>
<td style="text-align: right;"><span class="math inline">0.012</span></td>
<td style="text-align: right;">15</td>
</tr>
<tr>
<td style="text-align: left;">Frontier records, bw 730</td>
<td style="text-align: left;">unweighted</td>
<td style="text-align: right;"><span class="math inline">−0.00429</span></td>
<td style="text-align: right;"><span class="math inline">0.01592</span></td>
<td style="text-align: right;"><span class="math inline">−0.27</span></td>
<td style="text-align: right;"><span class="math inline">0.788</span></td>
<td style="text-align: right;">24</td>
</tr>
</tbody>
</table>

Notes: kinks in ECI points/day; HC1 standard errors against standard normal critical values. CI weights $`1/\sigma_i^2`$, $`\sigma_i = (\mathrm{ci}_{\mathrm{high}} -
\mathrm{ci}_{\mathrm{low}})/3.92`$. Placebo-cutoff grids (8 cutoffs at $`\pm90/180/270/360`$ days, bandwidth 540, triangular): all-models unweighted rejects 4/8 (true cutoff null in all cells); all-models CI-weighted rejects 3/8, same sign as the headline; frontier records rejects 5/8. Panel B donut and bandwidth rows are the perturbations of the frontier-records headline cell.

# Caveats and Conclusion

The caveats of record are these. First, this is a calendar-time RKD: model releases are not randomized around the cutoff, and HC1 treats them as independent (same-day multi-releases and lab-level clustering are ignored); the placebo-cutoff grid is the operative guard, and it is what both establishes power for the null and disqualifies the weighted “significant” cell. Second, data quality: 26 of 211 models (12.3%) carry Unverified/Speculative version-level Confidence flags — including the current \#1, GPT-5.6 Sol (161.77), and GPT-5 — and the confidence join matched only 189/211 models (22 unmatched treated as not-flagged); the high-confidence robustness subset reproduces every conclusion. Third, two models (Claude 3.5 Sonnet and GPT-5) have no published CI and enter the weighted runs at the median imputed $`\sigma`$; and the index itself is a modeled latent scale whose vintages revise as benchmarks are added , which is why the replication across vintages matters.

The verdict is an upgraded null with power. On twelve more months of data, the aggregate ECI trend shows no kink at o1: the unweighted estimate is a precise zero at a cutoff where the design demonstrably has power to reject (4/8 placebos reject elsewhere), the one new nominally significant result — the CI-weighted negative kink, significant in 18/18 cells — is correctly identified by triage as a smooth post-2024 deceleration of the precision-weighted index (same-sign larger placebos at $`+90`$/$`+180`$, absorbed by a local quadratic) rather than an o1-localized break, and the genuine frontier-records series remains a four-left-point non-design. Whatever the reasoning-model recipe did to AI capabilities, it did not bend the aggregate index’s calendar-time trend at the release date. Per the collection’s ranking specification, this result folds into the flagship paper’s ECI section as an appendix rather than standing as an independent mini-paper.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 ; estimation is deterministic (local weighted least squares), seed 0 was declared, and no stochastic step exists in the design. The underlying `eci_scores.csv` vintage is not committed to the repository; the run of record retains the derived panels, the 144-cell grid, the placebo grids, and the validation counts. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline estimates, the recomputed fit slopes, and the placebo rejection counts against the numbers of record before drawing.

<div class="thebibliography">

9

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552.

Ho, A., Denain, J.-S., Atanasov, D., Albanie, S., and Shah, R. (2025). A Rosetta Stone for AI benchmarks. arXiv:2512.00193.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

OpenAI (2024). OpenAI o1 System Card. arXiv:2412.16720.

Snell, C., Lee, J., Xu, K., and Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv:2408.03314.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
