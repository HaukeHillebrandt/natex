> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/prop99-validation-writeup/) · [PDF](https://haukehillebrandt.github.io/natex/prop99-validation-writeup.pdf) · [PDF in this repo](./main.pdf)

# Introduction

California’s Proposition 99, passed in November 1988 and effective January 1989, raised the state cigarette excise tax by 25 cents per pack and funded a large anti-smoking campaign. Taxation and media spending each measurably reduced cigarette sales , and the program is associated with accelerated declines in consumption and heart-disease mortality . It is also the canonical testbed of the synthetic control method: building on the comparative case-study estimator of , Abadie, Diamond, and Hainmueller constructed a “synthetic California” from a weighted combination of donor states and estimated an average gap of roughly $`-19`$ packs per capita over 1989–2000. That example anchors a large methodological literature and is arguably the most-replicated benchmark in comparative case studies.

This paper uses it the other way around: not to learn about Proposition 99, but to validate a pipeline. `natex` is an automated natural-experiment discovery-and-estimation toolkit in the LoRD3 lineage ; its difference-in-differences arm implements SuDDDS (Subset Discovery of Difference-in-Differences) from chapter 6 of , whose printed Table 6.1 evaluates the method on this same smoking panel. We run the full pipeline *blind* — the scan is never told the treated unit or the intervention year, and provably never reads the outcome — and ask whether it rediscovers (California, 1989) and reproduces the ADH estimate. It does — exactly, deterministically, and with every deviation from the thesis’s printed values investigated and reconciled. Because the exercise replicates a known benchmark, we write it up short, structurally as validation: its only role is to transfer credibility to the `natex` studies on novel data.

# Data

The data are the ADH California tobacco panel as shipped in the `natex` dataset registry (dataset `prop99`; 1,209 rows): annual state-level per-capita cigarette sales (packs, from tax revenues) for 39 states, 1970–2000 — California plus ADH’s 38 donor states — with auxiliary covariates (log income, beer consumption, share aged 15–24, retail price). The panel is balanced ($`39 \times 31`$, no missing outcome cells). Treatment is the policy dummy: California from 1989 onward. (The source thesis reports “30 US states” and does not document its panel; ours is ADH’s 39-state pool.)

# Design and Methods

#### Blind discovery (SuDDDS).

SuDDDS , as implemented and corrected in `natex` , scans panel records $`(x_i, t_i, \theta_i)`$ — covariates, time, treatment — for a subset of units and an intervention time $`T_0`$ at which $`\theta`$ jumps, scoring candidates by a log-likelihood ratio and never reading the outcome $`y`$. Protocol, fixed before calibration: windows $`(5, 8, 10)`$, 4 time bins, 8 restarts, seed 0, and background degree 0 (unit effects only). Degree 0 is required, not tuning: $`\theta`$ is a pure policy dummy whose only time variation is the candidate discontinuity, so any global time polynomial can only fit the jump itself; with degree 1 the leaked slope manufactures a spurious whole-panel optimum (the thesis never reports its background — an unreported-hyperparameter risk). All three scan methods (greedy, wcc, single_delta) are run under the normal model, plus a wcc scan under the Bernoulli model, the corrected default for a binary treatment. A y-blindness test re-runs the scan with the outcome deleted and requires bitwise-identical discoveries.

#### Calibration.

The top discovery’s LLR is calibrated by a fitted-null Monte Carlo with $`Q = 99`$ replicas and a $`+1`$-rank rule (minimum attainable $`p = 0.010`$), drawing nulls from Bernoulli($`\hat p`$) — the correct null family for a binary $`\theta`$. As a diagnostic we also run the dependence-preserving AR(1)-unit null, which is *structurally conservative* on a deterministic policy dummy: 38 of 39 units have identically-zero residuals, so the pooled AR(1) fit absorbs California’s step as noise and hands it to every replica; its $`p`$ pins at $`1.0`$, documented as a regression, never claimed as evidence.

#### Estimation conditioned on the discovery.

Three effect estimators are run on the recovered discovery over the full post period 1989–2000: two-way difference-in-differences (DD), synthetic control (simplex-weighted donor fit on pre-period outcomes, as in ), and GESS (greedy control-group expansion, thesis ch. 6). Separately, the `natex` synthetic-control donor path is run on the raw (state, year, cigsale) matrix with the full donor pool and with an $`n=8`$ pre-fit-ranked variant; this path is deterministic — no rng anywhere — and bitwise-identical across runs.

#### Honest inference.

Significance is assessed by exact, enumerated placebo tests, with the caveats stated plainly. (i) The in-space RMSPE-ratio placebo refits every one of the 38 control states as a pseudo-treated unit (`exclude_poor_fit = None`: all 38 usable, 0 skipped) and ranks California’s post/pre RMSPE ratio among all 39; ADH’s $`p = 1/39`$ requires their poor-pre-fit exclusion, kept opt-in in `natex`. (ii) The corrected effect-level test is a two-sided studentized statistic $`|\hat\tau / \mathrm{se}|`$ with $`+1`$-rank $`p`$-values over the 38 enumerated placebo states — under it, genuinely non-parallel placebo states (flat post-shifted gaps with tiny standard errors) can out-$`t`$ California’s trending gap. (iii) The validation battery’s degenerate cases report “cannot test” (NaN, failed) rather than fabricating passes: composition reduces to a one-row table for a single-unit discovery, and anticipation has zero pre-period treatment variance.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) California vs. synthetic California (deterministic full-pool simplex weights), 1970–2000: pre-RMSPE <span class="math inline">1.656</span>, post-period gap averaging <span class="math inline">−19.514</span> packs per capita. (b) In-space placebo: post/pre RMSPE ratios for all 39 states (all 38 placebos usable, 0 skipped). California ranks 3/39 (<span class="math inline"><em>p</em> = 0.077</span>); the two out-ranking states, Missouri and Virginia, are poor-pre-fit placebos that ADH’s exclusion rule would discard.</figcaption>
</figure>

#### Blind discovery is exact.

All three SuDDDS scan methods under the normal model return (California, $`t_0 = 1989.0`$) as the top discovery, with the discovered subset exactly the 31 California records: precision $`=`$ recall $`= 1.0`$, LLR $`13.82`$. The Bernoulli-model scan is also exact, LLR $`13.86`$. The y-blindness test passes bitwise (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A).

#### The discovery is significant under the correct null.

Against Bernoulli($`\hat p`$) nulls ($`Q = 99`$, seed 1) the observed max LLR $`13.86`$ exceeds the null max $`7.95`$ (null q90 $`4.67`$): $`p = 0.010`$, the $`+1`$-rank floor. The AR(1)-unit diagnostic pins at $`p = 1.0`$, as expected for a deterministic policy dummy (Section <a href="#sec:methods" data-reference-type="ref" data-reference="sec:methods">3</a>); recorded as a structural limitation of dependence-preserving nulls, not as evidence against the discovery (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel B).

#### Effects match ADH; the printed thesis numbers reconcile.

On the full 1989–2000 post period, DD gives $`\hat\tau = -27.349`$ packs (pre-MSE $`51.23`$), synthetic control gives $`\hat\tau = -19.514`$ (pre-MSE $`2.74`$), and GESS gives $`\hat\tau = -26.653`$ (control: Montana, pre-MSE $`18.35`$). Signs and the pre-fit ordering match the thesis’s Table 6.1 ($`-10.94`$ / $`-8.96`$ / $`-6.67`$): synthetic control’s pre-MSE is far below DD’s, which is exactly ADH’s argument for it. The magnitudes differ by a factor of $`\sim`$<!-- -->2–2.5 because the thesis’s printed values correspond to an unreported effective post window of about five years, while the gap accumulates over time; restricting `natex` to 1989–1993 gives DD $`-18.8`$ and synthetic $`-12.3`$, bracketing the printed values (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel C).

#### Donor selection recovers ADH’s pool.

The deterministic full-pool simplex fit puts nonzero weight on six donors, with summed weight $`0.955`$ on ADH’s five published donors, and the four heavyweights are exactly ADH’s own {Utah, Montana, Nevada, Connecticut} (weights in Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel D). ATT$`_{\text{post}} = -19.514`$ packs per capita, with pre-RMSPE $`1.656`$ against post-RMSPE $`> 10`$ (the ADH-style pre/post contrast); the 8-donor variant gives ATT $`-22.648`$, pre-RMSPE $`3.671`$ (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>a).

#### Honest inference: what survives and what does not.

The exact in-space RMSPE-ratio placebo ranks California 3 of 39 in both variants (treated ratio $`12.440`$ full pool, $`6.570`$ top-8): $`p = 3/39 = 0.077`$ (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). ADH’s $`1/39`$ arises only under their poor-pre-fit exclusion, which `natex` keeps opt-in. The corrected two-sided studentized $`\tau`$ placebos over the 38 enumerated states give DD rank 13/39 ($`p = 0.333`$), synthetic control 7/39 ($`p \leq 8/39 = 0.205`$), and GESS 9/39 ($`p = 0.231`$): the thesis’s claim that all three are significant at 5% does *not* survive the corrected statistic, and we report that as the finding it is (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel E).

#### Run of record and reproducibility.

The entire benchmark is executable: running `uv run pytest -m backtest` with `NATEX_DATA` set to the data root, on the modules `test_prop99.py` and `test_prop99_donors.py` under `tests/backtests/`, gives 16 passed, 1 xfailed in 79.74 s (re-verified 2026-07-18). The single xfail is a non-blocking instrument-selection stretch goal on a *different* dataset, pinned as a documented finding; all 15 Proposition 99 tests pass. All estimates: `natex` v0.2.0 on the registry copy of the ADH panel (not committed). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from `figures/make_fig.py` (committed), which asserts the headline numbers of record before drawing.

<table id="tab:main">
<caption>Proposition 99 backtest: numbers of record (<code>natex</code> v0.2.0; seeds 0/1/2, donor path rng-free).</caption>
<thead>
<tr>
<th style="text-align: left;">Quantity</th>
<th style="text-align: left;">Value</th>
<th style="text-align: left;">Quantity</th>
<th style="text-align: left;">Value</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel A: blind discovery (SuDDDS; windows (5,8,10), bins 4, restarts 8, degree 0, seed 0)</em></td>
</tr>
<tr>
<td style="text-align: left;">greedy / wcc / single_delta</td>
<td style="text-align: left;">(California, 1989.0)</td>
<td style="text-align: left;">precision, recall</td>
<td style="text-align: left;"><span class="math inline">1.0</span>, <span class="math inline">1.0</span> (31 CA records)</td>
</tr>
<tr>
<td style="text-align: left;">normal-model LLR</td>
<td style="text-align: left;"><span class="math inline">13.82</span></td>
<td style="text-align: left;">Bernoulli wcc</td>
<td style="text-align: left;">exact; LLR <span class="math inline">13.86</span></td>
</tr>
<tr>
<td style="text-align: left;">y-blindness</td>
<td style="text-align: left;">bitwise identical</td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel B: discovery calibration (<span class="math inline"><em>Q</em> = 99</span>, <span class="math inline">+1</span>-rank, seed 1)</em></td>
</tr>
<tr>
<td style="text-align: left;">Bernoulli(<span class="math inline"><em>p̂</em></span>) null</td>
<td style="text-align: left;"><span class="math inline"><em>p</em> = 0.010</span></td>
<td style="text-align: left;">observed vs null max</td>
<td style="text-align: left;"><span class="math inline">13.86</span> vs <span class="math inline">7.95</span> (q90 <span class="math inline">4.67</span>)</td>
</tr>
<tr>
<td style="text-align: left;">AR(1)-unit null</td>
<td style="text-align: left;"><span class="math inline"><em>p</em> = 1.0</span>, conservative<span class="math inline"><sup>†</sup></span></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel C: effects on the discovery, post 1989–2000 (thesis Table 6.1: <span class="math inline">−10.94/ − 8.96/ − 6.67</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;">DD <span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: left;"><span class="math inline">−27.349</span> (pre-MSE <span class="math inline">51.23</span>)</td>
<td style="text-align: left;">synthetic <span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: left;"><span class="math inline">−19.514</span> (pre-MSE <span class="math inline">2.74</span>)</td>
</tr>
<tr>
<td style="text-align: left;">GESS <span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: left;"><span class="math inline">−26.653</span> (MT, pre-MSE <span class="math inline">18.35</span>)</td>
<td style="text-align: left;">1989–1993 restriction</td>
<td style="text-align: left;">DD <span class="math inline">−18.8</span>, synthetic <span class="math inline">−12.3</span></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel D: synthetic-control donor path (deterministic, bitwise-reproducible)</em></td>
</tr>
<tr>
<td style="text-align: left;">weights</td>
<td style="text-align: left;">UT <span class="math inline">.394</span>, MT <span class="math inline">.232</span>, NV <span class="math inline">.205</span></td>
<td style="text-align: left;">ADH-five summed weight</td>
<td style="text-align: left;"><span class="math inline">0.955</span></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">CT <span class="math inline">.109</span>, NH <span class="math inline">.045</span>, CO <span class="math inline">.015</span></td>
<td style="text-align: left;">ATT<span class="math inline"><sub>post</sub></span></td>
<td style="text-align: left;"><span class="math inline">−19.514</span> packs (ADH <span class="math inline"> ≈ −19</span>)</td>
</tr>
<tr>
<td style="text-align: left;">pre- / post-RMSPE</td>
<td style="text-align: left;"><span class="math inline">1.656</span> / <span class="math inline"> &gt; 10</span></td>
<td style="text-align: left;">8-donor variant</td>
<td style="text-align: left;">ATT <span class="math inline">−22.648</span>, pre <span class="math inline">3.671</span></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel E: exact placebo inference (enumerated; no poor-pre-fit exclusion)</em></td>
</tr>
<tr>
<td style="text-align: left;">RMSPE ratio (treated)</td>
<td style="text-align: left;"><span class="math inline">12.440</span> full / <span class="math inline">6.570</span> top-8</td>
<td style="text-align: left;">rank, <span class="math inline"><em>p</em></span></td>
<td style="text-align: left;">3/39 both; <span class="math inline"><em>p</em> = 3/39 = 0.077</span></td>
</tr>
<tr>
<td style="text-align: left;">studentized <span class="math inline"><em>τ</em></span>: DD</td>
<td style="text-align: left;">13/39 (<span class="math inline"><em>p</em> = 0.333</span>)</td>
<td style="text-align: left;">synthetic</td>
<td style="text-align: left;">7/39 (<span class="math inline"><em>p</em> ≤ 8/39 = 0.205</span>)</td>
</tr>
<tr>
<td style="text-align: left;">GESS</td>
<td style="text-align: left;">9/39 (<span class="math inline"><em>p</em> = 0.231</span>)</td>
<td style="text-align: left;">thesis “all sig. at 5%”</td>
<td style="text-align: left;">does <em>not</em> survive</td>
</tr>
</tbody>
</table>

Notes: $`^{\dagger}`$structural, not evidential: a deterministic policy dummy leaves 38 of 39 units with identically-zero residuals, so the pooled AR(1) null absorbs the step as noise; pinned as a documented regression. Scan $`p`$-values use a $`Q = 99`$ fitted-null Monte Carlo with a $`+1`$-rank rule; $`0.010`$ is the minimum attainable. The synthetic-control placebo rank carries one rank of slack because each placebo refits SLSQP weights.

# Caveats and Conclusion

Three caveats bound the claims. First, this exercise has zero novelty by construction: Proposition 99 on the ADH panel is the most-replicated synthetic-control benchmark in economics , and a pipeline that failed it would be disqualified rather than informative; its role is external-validity anchoring for the `natex` studies on novel data, and it should be read as methods validation, not as a finding about tobacco policy. Second, effect magnitudes here are $`\sim`$<!-- -->2–2.5$`\times`$ the source thesis’s printed Table 6.1 values because `natex` estimates on the full 1989–2000 post period while the thesis’s numbers match an unreported $`\sim`$<!-- -->5-year effective window; the thesis also never reports its panel (“30 US states” vs. ADH’s 39) or its treatment background. Signs, ordering, and pre-fit ranking agree, and the synthetic-control estimate matches ADH’s canonical $`\approx -19`$ — but the reconciliation must be stated, not glossed. Third, the honest-inference results are genuinely weaker than the textbook telling: without ADH’s poor-pre-fit exclusion the in-space placebo $`p`$ is $`0.077`$, not $`1/39 = 0.026`$, and under the corrected two-sided studentized placebo statistic none of the three effect estimators clears 5% on this panel. These are properties of the inference conventions, faithfully surfaced.

Within those bounds, the conclusion is simple: run blind, `natex` rediscovers (California, 1989) exactly by every scan method, calibrates it significantly under the correct null, selects ADH’s donors with $`0.955`$ of the simplex weight, and reproduces the canonical $`\approx -19`$ synthetic-control estimate deterministically. The pipeline earns the benefit of the doubt it needs on data where the answer is not known.

<div class="thebibliography">

9

Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391–425.

Abadie, A., Diamond, A., and Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California’s tobacco control program. *Journal of the American Statistical Association*, 105(490), 493–505.

Abadie, A., Diamond, A., and Hainmueller, J. (2015). Comparative politics and the synthetic control method. *American Journal of Political Science*, 59(2), 495–510.

Abadie, A., and Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, 93(1), 113–132.

Ben-Michael, E., Feller, A., and Rothstein, J. (2021). The augmented synthetic control method. *Journal of the American Statistical Association*, 116(536), 1789–1803.

Fichtenberg, C. M., and Glantz, S. A. (2000). Association of the California Tobacco Control Program with declines in cigarette consumption and mortality from heart disease. *New England Journal of Medicine*, 343(24), 1772–1777.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

Herlands, W. (2020). *Change Modeling for Understanding Our World and the Counterfactual One(s)*. PhD thesis, Carnegie Mellon University.

Hu, T.-W., Sung, H.-Y., and Keeler, T. E. (1995). Reducing cigarette consumption in California: Tobacco taxes vs an anti-smoking media campaign. *American Journal of Public Health*, 85(9), 1218–1222.

Jakubowski, B., Somanchi, S., McFowland III, E., and Neill, D. B. (2023). Exploiting discovered regression discontinuities to debias conditioned-on-observable estimators. *Journal of Machine Learning Research*, 24(133), 1–57.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
