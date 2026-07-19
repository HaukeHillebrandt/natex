> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/chinchilla-writeup/) · [PDF](https://haukehillebrandt.github.io/natex/chinchilla-writeup.pdf) · [PDF in this repo](./main.pdf)

# Introduction

On 29 March 2022, Hoffmann et al. posted the “Chinchilla” result: for compute-optimal training, model size and training tokens should scale in equal proportion, implying that the then-standard practice — descended from the earlier scaling-law fits of — left frontier models substantially undertrained on data . Release practice responded within months, and by 2023 the industry had moved past compute-optimality into deliberate over-training of small models for inference economy, exemplified by LLaMA’s trillion-token training runs and rationalized by inference-adjusted scaling laws . Chinchilla itself has since been re-examined: replicate its parametric fit and find inconsistencies in the reported estimates, while the broader quantitative history of training-run inputs is documented by and maintained in Epoch AI’s notable-models dataset .

This gives a rare, clean testbed for *automated natural-experiment discovery*: a panel where the event, its date, its sign, and its approximate magnitude are known ex ante. This paper runs the `natex` toolkit’s LoRD3 scan on tokens per parameter over calendar time, blind to the event date, and asks: does the scanner find Chinchilla? The answer is no — and the anatomy of the miss is the contribution. Event-dated diffusion processes enter publication data as ramps, and a local level-break statistic rewards sharp edges, not ramps; dating the cause is an interrupted-time-series problem , not a discontinuity-scan problem.

# Data

The panel is a frozen extract of Epoch AI’s notable-models database : 549 language models released 2019-01 through 2024-12 with non-missing parameter and training-token counts. Columns: release date (the forcing variable, as days since 2019-01-01), $`\log_{10}`$ tokens per parameter (the treatment), $`\log_{10}`$ parameters (the outcome), $`\log_{10}`$ training compute (21% missing) and an open-weights flag (3.6% missing), both excluded from the scan to avoid listwise deletion. The known truth: Chinchilla was posted 2022-03-29 (day $`\approx 1183`$); across that date the global median tokens per parameter moves from $`1.69`$ to $`29.17`$ ($`n = 185`$ pre, $`364`$ post), a median-based shift of $`+1.24`$ $`\log_{10}`$ units ($`+1.12`$ on group means, $`2.2 \to 29`$; Mann–Whitney $`p = 1.6 \times 10^{-7}`$ on an earlier extract, medians $`2.2 \to 29.1`$, $`n = 133/307`$). At monthly resolution the shift is a ramp, not a step: window medians run $`1.7`$ (pre-April 2022) $`\to 5.7`$ (mid-April to mid-May) $`\to`$ $`{\approx}28`$ (mid-May to mid-July 2022).

# Design and Methods

#### LoRD3 continuous-treatment scan.

Following as implemented in `natex` v0.2.0 , the scan fits a background model of the treatment on the forcing variable (OLS, degree 1; degree 2 as robustness), then scores every $`k`$-nearest-neighbor neighborhood ($`k = 50`$) of every data point for the best split of its members into two half-spaces, using a normal likelihood ratio on background residuals with locally estimated variances. The top discovery is the neighborhood-split with the maximum LLR; its $`p`$-value comes from a fitted-null Monte Carlo with $`q = 99`$ replicas and a $`+1`$-rank rule, so the minimum attainable $`p`$ is $`0.010`$ — reported throughout as $`p \leq 0.01`$, never as an exact value. The specification is treatment $`= \log_{10}`$ tokens per parameter, forcing $`=`$ days since 2019-01-01 (forcing influence $`1.0`$), outcome $`= \log_{10}`$ parameters, no covariates; pipeline `natex study` $`\to`$ `natex discover --k 50 --q 99 --seed 0`, with localization detail and robustness via the Python API, seed 0 throughout.

#### Validation battery, stated as run.

The battery’s nulls and refusals are reported, not patched over. (i) Randomization: $`p = 0.010`$ at both background degrees — the $`q=99`$ floor. (ii) Placebo covariates: `placebo_passed = True` but *vacuous* — `placebo_holm = {}`, because the design has no non-forcing covariates to test. (iii) Density: $`p = 0.0041`$, a *fail*; this is expected and uninformative here, because the forcing is calendar time with strongly nonstationary publication intensity, and McCrary-style manipulation logic does not apply to a time forcing — the test is retained as a falsification check only. (iv) Post-selection effects on $`\log_{10}`$ parameters (2SLS and Wald at the discovered cutoff) are *advisory*: estimated on the discovery sample with no honest split and instrumented by a partly artifactual break, they must not be read as causal scaling-law parameters. (v) An independent kink-design audit of the same panel (a difference-in-slopes design ) cross-checks any slope-change reading.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) <span class="math inline">log<sub>10</sub></span> tokens per parameter for all 549 models, with the monthly median (line), the Chinchilla posting date (dashed), the scan’s top discovery (solid), and the seven verified fine-tune token-counting artifact rows (open triangles; Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel C). (b) LoRD3 scan profile: each candidate neighborhood’s LLR at its center date (<span class="math inline"><em>k</em> = 50</span>, degree 1, <span class="math inline"><em>n</em> = 547</span> candidates). The top discovery (filled marker, 2022-09-22, LLR <span class="math inline">8.69</span>, <span class="math inline"><em>p</em> ≤ 0.01</span>) sits at the Sep/Oct 2022 artifact edge; the best candidate within <span class="math inline">±45</span> days of the truth (open marker, 2022-04-15) scores LLR <span class="math inline">1.39</span>, rank 158/547. The 2023-06/07 cluster is the onset of the deliberately over-trained era.</figcaption>
</figure>

#### The discovery.

The scan’s top discovery (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b) is centered on day 1360 $`=`$ 2022-09-22 (split normal $`-1`$ along the forcing axis): LLR $`= 8.687`$ ($`8.687416`$ unrounded in the run report), scan $`p = 0.010`$ — the $`q=99`$ Monte Carlo floor, i.e. $`p \leq 0.01`$ — over $`547`$ candidate neighborhoods from $`n = 549`$ rows. A degree-2 background reproduces the same top center with LLR $`8.836`$ (randomization $`p = 0.01`$; null max LLRs $`1.8`$–$`7.6`$ against observed $`8.69`$/$`8.84`$), and $`k = 60`$ moves it only to 2022-08-15; but $`k = 30`$/$`40`$ latch onto a 2020-10 composition boundary instead (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel B), so localization is stable only for $`k \geq 50`$.

#### The known truth is not localized.

Within $`\pm 45`$ days of 2022-03-29 the best candidate neighborhood is centered 2022-04-15 with LLR $`1.39`$ — rank 158 of 547. The reason is visible in Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>a: at daily resolution the Chinchilla shift enters the published record as a 2–3 month adoption ramp, and a local level-break statistic rewards sharp edges, not ramps.

#### What the discovered break actually is.

The top neighborhood splits exactly at the Sep/Oct 2022 boundary into group A (2022-06-21 to 2022-09-22, $`n=25`$, median tokens/param $`13.6`$ — the post-Chinchilla adoption wave) versus group B (2022-10-03 to 2022-12-22, $`n=25`$, median $`0.4`$): a *downward* local group-mean contrast of $`1.63`$ $`\log_{10}`$ units ($`\times 43`$), against the known *upward* global shift of $`+1.12`$ $`\log_{10}`$ ($`+1.24`$ median-based). Local-linear first-stage fits jump $`-1.7`$ to $`-2.0`$ $`\log_{10}`$ at bandwidths of 120–240 days. Group B is dominated by fine-tuned and continued-pretraining releases whose Epoch token field records only fine-tuning tokens over full base parameters (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel C) plus small academic models. The top discovery is therefore the adoption wave’s downstream edge colliding with a token-accounting artifact, $`{\approx}6`$ months after the event and with the opposite sign. The secondary discovered cluster (2023-06-13 to 2023-07-06, LLR $`6.7`$–$`8.5`$) is the onset of the deliberately over-trained era ; the December-2022 monthly median is already $`148`$ tokens/param, and the median of the top-25 centers is day 1371 $`=`$ 2022-10-03.

#### Effects and cross-checks, all null or advisory.

Post-selection effects on $`\log_{10}`$ parameters at the discovered cutoff (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel D): the 2SLS estimate is weak-instrument (first-stage $`t = 1.89`$; CI $`[-0.747, 2.781]`$); the Wald estimate is $`0.557`$ (CI $`[0.053, 1.062]`$, first-stage $`t = 4.01`$). Both are discovery-sample estimates with no honest split, instrumented by a partly artifactual break — diagnostics, not causal scaling-law parameters. The independent kink audit *rejects* a Chinchilla regime change read as a slope change: a sign-unstable null with $`t`$ from $`-0.91`$ to $`+1.48`$ across bandwidths.

<table id="tab:main">
<caption>Chinchilla scan: numbers of record (<code>natex</code> v0.2.0, seed 0).</caption>
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
<td colspan="4" style="text-align: left;"><em>Panel A: top discovery (<span class="math inline"><em>k</em> = 50</span>, <span class="math inline"><em>q</em> = 99</span>, degree 1; <span class="math inline"><em>n</em> = 549</span>, 547 candidates)</em></td>
</tr>
<tr>
<td style="text-align: left;">center</td>
<td style="text-align: left;">2022-09-22 (day 1360)</td>
<td style="text-align: left;">LLR</td>
<td style="text-align: left;"><span class="math inline">8.687</span></td>
</tr>
<tr>
<td style="text-align: left;">scan <span class="math inline"><em>p</em></span></td>
<td style="text-align: left;"><span class="math inline">0.010</span> = floor; <span class="math inline"><em>p</em> ≤ 0.01</span></td>
<td style="text-align: left;">local contrast</td>
<td style="text-align: left;"><span class="math inline">−1.63 log<sub>10</sub></span> (<span class="math inline">×43</span>, down)</td>
</tr>
<tr>
<td style="text-align: left;">degree-2 center</td>
<td style="text-align: left;">2022-09-22, LLR <span class="math inline">8.836</span></td>
<td style="text-align: left;">first stage</td>
<td style="text-align: left;"><span class="math inline">−1.7</span> to <span class="math inline">−2.0 log<sub>10</sub></span> (bw 120–240d)</td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel B: <span class="math inline"><em>k</em></span>-sensitivity and truth localization</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>k</em> = 30</span></td>
<td style="text-align: left;">2020-09-29 (LLR <span class="math inline">7.2</span>)</td>
<td style="text-align: left;"><span class="math inline"><em>k</em> = 40</span></td>
<td style="text-align: left;">2020-10-21 (LLR <span class="math inline">9.0</span>)</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>k</em> = 50</span></td>
<td style="text-align: left;">2022-09-22 (LLR <span class="math inline">8.7</span>)</td>
<td style="text-align: left;"><span class="math inline"><em>k</em> = 60</span></td>
<td style="text-align: left;">2022-08-15 (LLR <span class="math inline">9.8</span>)</td>
</tr>
<tr>
<td style="text-align: left;">truth <span class="math inline">±45</span>d best</td>
<td style="text-align: left;">2022-04-15, LLR <span class="math inline">1.39</span></td>
<td style="text-align: left;">truth rank</td>
<td style="text-align: left;">158 / 547</td>
</tr>
<tr>
<td style="text-align: left;">global shift (true)</td>
<td style="text-align: left;"><span class="math inline">+1.12 log<sub>10</sub></span> (<span class="math inline">2.2 → 29</span>, up)</td>
<td style="text-align: left;">medians</td>
<td style="text-align: left;"><span class="math inline">1.69 → 29.17</span> (<span class="math inline"><em>n</em> = 185/364</span>)</td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel C: verified fine-tune artifact rows (tokens/param; group B median <span class="math inline">0.4</span> vs. A <span class="math inline">13.6</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;">Flan-PaLM 540B</td>
<td style="text-align: left;"><span class="math inline">0.0026</span></td>
<td style="text-align: left;">U-PaLM</td>
<td style="text-align: left;"><span class="math inline">0.0024</span></td>
</tr>
<tr>
<td style="text-align: left;">LMSI-PaLM</td>
<td style="text-align: left;"><span class="math inline">3.6 × 10<sup>−6</sup></span></td>
<td style="text-align: left;">OPT-IML 175B</td>
<td style="text-align: left;"><span class="math inline">0.011</span></td>
</tr>
<tr>
<td style="text-align: left;">Tk-Instruct</td>
<td style="text-align: left;"><span class="math inline">0.095</span></td>
<td style="text-align: left;">BLOOMZ-176B</td>
<td style="text-align: left;"><span class="math inline">0.114</span></td>
</tr>
<tr>
<td style="text-align: left;">mT0-13B</td>
<td style="text-align: left;"><span class="math inline">1.54</span></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>Panel D: diagnostics and post-selection effects (advisory)</em></td>
</tr>
<tr>
<td style="text-align: left;">randomization</td>
<td style="text-align: left;"><span class="math inline"><em>p</em> = 0.010</span> (deg. 1 and 2)</td>
<td style="text-align: left;">placebo</td>
<td style="text-align: left;">passed, <em>vacuous</em> (no covariates)</td>
</tr>
<tr>
<td style="text-align: left;">density</td>
<td style="text-align: left;"><span class="math inline"><em>p</em> = 0.0041</span>, fails<span class="math inline"><sup>†</sup></span></td>
<td style="text-align: left;">kink audit</td>
<td style="text-align: left;">rejected; <span class="math inline"><em>t</em> ∈ [−0.91, +1.48]</span></td>
</tr>
<tr>
<td style="text-align: left;">2SLS <span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: left;"><span class="math inline">1.017</span> (se <span class="math inline">0.900</span>), weak</td>
<td style="text-align: left;">CI</td>
<td style="text-align: left;"><span class="math inline">[−0.747, 2.781]</span>, 1st-st. <span class="math inline"><em>t</em> = 1.89</span></td>
</tr>
<tr>
<td style="text-align: left;">Wald <span class="math inline"><em>τ̂</em></span></td>
<td style="text-align: left;"><span class="math inline">0.557</span> (se <span class="math inline">0.257</span>)</td>
<td style="text-align: left;">CI</td>
<td style="text-align: left;"><span class="math inline">[0.053, 1.062]</span>, 1st-st. <span class="math inline"><em>t</em> = 4.01</span></td>
</tr>
</tbody>
</table>

Notes: $`^{\dagger}`$expected and uninformative for a calendar-time forcing with nonstationary publication intensity (McCrary manipulation logic does not apply). Scan $`p`$-values from a $`q=99`$ fitted-null Monte Carlo with a $`+1`$-rank rule; $`0.010`$ is the minimum attainable. Effects are estimated on the discovery sample with no honest split. Artifact rows verified in the source panel: the token field records fine-tuning tokens only, over full base parameters.

# Caveats and Conclusion

Five caveats bound the claims. First, the value of this exercise is purely methodological and diagnostic: no causal adoption claim is licensed; the scan is a regime-edge detector, and dating the cause of a diffusion process needs an interrupted-time-series or difference-in-differences framing plus publication-lag domain knowledge. Second, the scan $`p = 0.010`$ is the minimum attainable with $`q = 99`$ replicas and must be reported as $`p \leq 0.01`$, not as an exact value. Third, the density test fails ($`p = 0.0041`$) but is uninformative for a calendar-time forcing with nonstationary publication intensity, and the placebo pass is vacuous (`placebo_holm = {}`; no non-forcing covariates to test). Fourth, the post-selection 2SLS/Wald effects are estimated on the discovery sample with no honest split and are instrumented by a partly artifactual break; they must not be read as causal scaling-law parameters. Fifth, “robust to $`k`$” holds only for $`k \geq 50`$: at $`k = 30`$/$`40`$ the scan latches onto a 2020-10 composition boundary (tiny academic LMs against BERT-era corpus models), so the robustness claim must be stated conditionally.

The verdict is an artifact — an honest miss, correctly diagnosed: the scan found real, significant structure whose two regimes bracket the truth, but it dated the adoption wave’s sharp downstream *edge*, not the event, which entered the record as a ramp. Two lessons transport beyond this dataset. For method builders: scan batteries aimed at event-dated diffusion treatments need a ramp/slope statistic — a kink-style scan in the sense of — alongside the level-break statistic, or announcements will systematically lose to their own adoption edges. For users of the Epoch panel : fine-tuned and continued-pretraining rows manufacture a spurious low-tokens-per-parameter cluster in late 2022 and must be filtered before tokens-per-parameter data are used for scaling-law or event-study claims .

#### Reproducibility.

All estimates: `natex` v0.2.0 , seed 0, on a frozen extract of the Epoch panel (not committed). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from `figures/make_fig.py` (committed), which asserts the headline numbers of record before drawing.

<div class="thebibliography">

9

Lopez Bernal, J., Cummins, S., and Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 46(1), 348–355.

Besiroglu, T., Erdil, E., Barnett, M., and You, J. (2024). Chinchilla scaling: A replication attempt. arXiv:2404.10102.

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Epoch AI (2024). Data on notable AI models. <https://epoch.ai/data/notable-ai-models>

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training compute-optimal large language models. In *Advances in Neural Information Processing Systems*, 35 (NeurIPS 2022). arXiv:2203.15556, posted 29 March 2022.

Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.

McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

Sardana, N., Portes, J., Doubov, S., and Frankle, J. (2024). Beyond Chinchilla-optimal: Accounting for inference in language model scaling laws. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235, 43445–43460.

Sevilla, J., Heim, L., Ho, A., Besiroglu, T., Hobbhahn, M., and Villalobos, P. (2022). Compute trends across three eras of machine learning. In *2022 International Joint Conference on Neural Networks (IJCNN)*.

Touvron, H., et al. (2023). LLaMA: Open and efficient foundation language models. arXiv:2302.13971.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
