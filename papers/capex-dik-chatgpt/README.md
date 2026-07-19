> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/capex-dik-chatgpt/) · [PDF in this repo](./main.pdf)

# Introduction

OpenAI released ChatGPT on 30 November 2022. Within two years the capital expenditure of the four large US “hyperscalers” — Microsoft, Alphabet, Amazon, and Meta — had become a macroeconomic quantity in its own right, central to debates about how much aggregate investment the current AI wave can move . Financial markets repriced AI exposure almost immediately: construct firm-level workforce exposures to generative AI and show that an exposure-sorted long-short portfolio earned roughly 5% in the two weeks after the release. A natural real-economy question follows: did the release bend the hyperscalers’ *investment* trend — the growth rate of measured capital expenditure — or was the buildout already accelerating?

Any calendar-time answer must contend with a crowded event window. Eight weeks before ChatGPT, on 7 October 2022, the Bureau of Industry and Security announced sweeping export controls on advanced AI chips and semiconductor manufacturing equipment, a policy landmark in its own right ; any quarterly series treats the two events as nearly the same date.

The estimand — a change in trend slope for one group relative to another at a known date — is the difference-in-kinks (DiK) design formalized by , building on the regression kink design of . Kink estimates on time-series running variables are known to produce spuriously significant slope changes, which is why placebo distributions at non-event dates are the appropriate yardstick ; and conventional robust standard errors are badly oversized on serially correlated aggregates . This paper applies the `natex` toolkit’s DiK estimator to a two-group quarterly capex panel built from SEC EDGAR filings, with ChatGPT as the candidate cutoff and a mandatory falsification battery — which converts a nominally overwhelming event-study result into an explicit non-attribution.

# Data

Quarterly capital expenditure is reconstructed from SEC EDGAR XBRL `companyconcept` facts, converting fiscal year-to-date values to clean quarters by same-start differencing with remainder assignment across tag transitions. The tag is `PaymentsToAcquirePropertyPlantAndEquipment`; Amazon additionally reports under `PaymentsToAcquireProductiveAssets`. The treated series is the big-4 aggregate over 46 quarters, 2014Q4–2026Q1, rising from \$6.8bn to \$129.8bn per quarter. The rebuilt aggregate matches the frozen reference series of the run of record on all 45 common quarters (maximum difference \$0.000000bn) and its processed $`\log_{10}`$ counterpart on all 46 (maximum difference $`5\times10^{-5}`$).

The comparison series aggregates six large non-AI capital-intensive firms: Walmart, ExxonMobil, AT&T, Verizon, Union Pacific, and Home Depot, fetched through the identical EDGAR pipeline. Firm-specific tag quirks (AT&T, Verizon, and Home Depot report quarterly capex under `ProductiveAssets`-family tags; Walmart and Home Depot have January-31 fiscal year ends mapped to the nearest calendar quarter) are resolved against the live EDGAR records, and 2023 calendar-year sums match published figures for all six firms (AT&T \$17.85bn, Verizon \$18.77bn, ExxonMobil \$21.92bn, Walmart \$20.61bn, Union Pacific \$3.61bn, Home Depot \$3.23bn).

The outcome is $`\log_{10}`$ of group capex in \$bn per quarter, so slopes are in dex/yr (one dex is a factor of ten). The running variable is the quarter midpoint in fractional years; the ChatGPT cutoff is $`t_0=2022.913`$ (2022-11-30). A donut of $`0.13`$ years ($`\approx`$ one quarter) removes 2022Q4, the quarter that straddles the release.

# Design and Methods

#### Group difference-in-kinks.

Following , the DiK estimand is the change in the outcome’s slope in calendar time at the cutoff for the treated group *minus* the same slope change for the comparison group, estimated by local linear weighted regression on each side of the cutoff within each group (`natex` v0.2.0 ; the group indicator enters the CLI’s DiK period dimension with a unit policy-kink change, so $`\hat\tau = (\text{kink})_{\text{big-4}} - (\text{kink})_{\text{control}}`$ in dex/yr). The primary specification uses bandwidth 3.0 years, triangular kernel, the one-quarter donut, and HC1 standard errors; $`n=46`$ quarter-cells (11 pre-cutoff and 12 post-cutoff quarters per group).

#### Honest-inference caveats, stated as run.

The estimator’s own caveats apply verbatim: the cutoff and bandwidth are user-specified, so bandwidth, donut, and placebo-cutoff sensitivity must be reported; the reported interval is conventional local-polynomial inference and may retain smoothing bias; and causal interpretation requires parallel changes in non-policy slope kinks and a time-stable marginal response at the cutoff. HC1/CR1 inference additionally assumes serial independence, which these aggregates violate: residual lag-1 autocorrelation in the primary fit is $`0.657`$ on the control pre-cutoff side and $`0.367`$ on the big-4 pre-cutoff side, precisely the setting in which nominal robust standard errors are known to be understated .

#### Falsification battery.

\(i\) an 18-cell specification grid (bandwidth $`\in\{2,3,4\}`$ years $`\times`$ donut $`\in\{0,0.13,0.25\}`$ $`\times`$ {triangular, uniform}); (ii) a placebo cutoff at the BIS export controls (2022-10-07, $`t=2022.767`$); (iii) a clean *pre-period* placebo-cutoff grid — sample restricted to $`t<2022.7`$, 15 cutoffs 2017.0–2020.5, bandwidth 2.0, donut $`0.13`$ — in the spirit of the permutation logic of , yielding a placebo-calibrated $`p`$-value for the main effect at the matched bandwidth; (iv) a cutoff localization scan over 2021.5–2024.75; (v) per-firm staggered DiKs of each of the ten firms against the control aggregate; and (vi) robustness checks: CR1 clustering by year and by quarter, quarter-of-year dummies, leave-one-out and no-retail control aggregates, a role-swap sanity check, and a degree-2 polynomial.

#### Automated pipeline, stated as run.

The `natex` survey pipeline (`natex survey --seed 0`) was recorded for completeness: its automated kink family pools both groups on the auto-picked linear capex outcome and therefore cannot express the group-DiK aliasing; it returns null (Holm $`p=0.24`$). The remaining families were skipped, needs-input, or failed as expected on a 92-row two-unit panel. The manual DiK runs above are the primary evidence.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Quarterly capital expenditure ($bn, log scale) of the big-4 hyperscaler aggregate and the non-AI capital-intensive control aggregate, 2014Q4–2026Q1, with the BIS export-control (2022-10-07, dashed) and ChatGPT (2022-11-30, solid) dates marked. (b) Group DiK estimates with 95% HC1 confidence intervals at 15 pre-period non-event placebo cutoffs (gray; pre-only sample, <span class="math inline"><em>t</em> &lt; 2022.7</span>) and at the ChatGPT cutoff (vermilion), all at the matched bandwidth 2.0 yr, triangular kernel, one-quarter donut; filled markers are nominally significant at the 5% level. Seven of 15 placebos reject, and the largest placebo <span class="math inline">|<em>z</em>|</span> exceeds the ChatGPT <span class="math inline">|<em>z</em>|</span>.</figcaption>
</figure>

#### The bend is real …

The primary group DiK at ChatGPT (Table <a href="#tab:dik" data-reference-type="ref" data-reference="tab:dik">1</a>, first row) is $`\hat\tau=+0.1396`$ dex/yr (HC1 se $`0.0302`$, 95% CI $`[0.0804, 0.1988]`$, $`z=4.62`$, nominal $`p=3.85\times10^{-6}`$). It decomposes into a big-4 kink of $`+0.0878`$ dex/yr (slope $`0.0999\to0.1876`$; the aggregate’s growth roughly doubles, to a pace that multiplies capex by ten in about 5.3 years) and a control kink of $`-0.0518`$ ($`0.0811\to0.0293`$). The estimate is positive in every cell of the 18-cell grid ($`+0.0493`$ to $`+0.1951`$, all nominal $`p<0.05`$; the uniform kernel roughly halves $`\hat\tau`$ relative to the triangular at bandwidths $`\geq3`$). The point estimate is essentially invariant to CR1 clustering by year (se $`0.0313`$) or quarter (se $`0.0278`$), quarter-of-year dummies ($`+0.1396`$, se $`0.0296`$), leave-one-out control aggregates ($`+0.1329`$ to $`+0.1515`$), and a no-retail control of ExxonMobil, AT&T, Verizon, and Union Pacific ($`+0.1453`$, se $`0.0361`$); the role-swap check returns exactly $`-0.1396`$; the degree-2 estimate ($`+0.3022`$, se $`0.0914`$) shows curvature sensitivity in magnitude but not sign. Per firm against the control aggregate: Microsoft $`+0.1710`$ (se $`0.0366`$), Alphabet $`+0.2061`$ (se $`0.0349`$), Amazon $`+0.1375`$ (se $`0.0512`$), Meta $`+0.0308`$ (se $`0.0518`$, $`p=0.55`$ — a true null: Meta’s surge came in 2024, after its 2022–23 pullback); control firms’ own kinks are all mildly negative ($`-0.022`$ to $`-0.090`$). A localization scan (bandwidth 2.0, no donut) finds the DiK crossing zero near $`t=2022.3`$, maximizing at $`t=2023.25`$ ($`+0.2213`$, se $`0.0440`$), staying above $`+0.19`$ over 2023.0–2023.75, and decaying to $`\approx0`$ by 2024.5 — consistent with a lagged capex response to a late-2022 shock.

#### … but attribution to ChatGPT fails.

Three findings block a causal reading. *First*, the pre-period placebo battery fails decisively: at 15 non-event cutoffs in 2017.0–2020.5, the identical specification rejects at nominal 5% seven times (empirical size $`0.467`$; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). Calibrated against this placebo distribution, the ChatGPT estimate at the matched bandwidth ($`+0.1778`$, $`z=4.35`$) has $`p_{|z|}=(1+1)/(15+1)=0.125`$ and $`p_{|\tau|}=(3+1)/16=0.250`$; the largest placebo $`|z|`$ is $`5.11`$ (at 2019.75) and the largest placebo $`|\hat\tau|`$ is $`0.284`$ (at 2019.5) — both exceeding the ChatGPT values. *Second*, the full-sample kink at the non-event date 2021.5 is $`-0.3135`$ (se $`0.0452`$, $`|z|=6.9`$), larger in magnitude and nominal significance than the main effect: these smooth aggregates generate huge nominal kinks where nothing happened. *Third*, the cutoff is confounded: a DiK at the BIS export-control date eight weeks earlier returns $`+0.1253`$ (se $`0.0317`$, $`z=3.95`$), statistically indistinguishable from the ChatGPT estimate — quarterly data cannot separate the two events (or any other late-2022 candidate shock).

<table id="tab:dik">
<caption>Group difference-in-kinks estimates, <span class="math inline">log<sub>10</sub></span> capex (dex/yr).</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline">|<em>z</em>|</span></th>
<th style="text-align: center;">sig. 5%</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: ChatGPT cutoff (<span class="math inline"><em>t</em><sub>0</sub> = 2022.913</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 3, donut (primary)</td>
<td style="text-align: right;"><span class="math inline">+0.1396</span></td>
<td style="text-align: right;"><span class="math inline">0.0302</span></td>
<td style="text-align: right;"><span class="math inline">4.62</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">triangular, bw 2, donut (placebo-matched)</td>
<td style="text-align: right;"><span class="math inline">+0.1778</span></td>
<td style="text-align: right;"><span class="math inline">0.0409</span></td>
<td style="text-align: right;"><span class="math inline">4.35</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">uniform, bw 4, donut (grid minimum)</td>
<td style="text-align: right;"><span class="math inline">+0.0493</span></td>
<td style="text-align: right;"><span class="math inline">0.0210</span></td>
<td style="text-align: right;"><span class="math inline">2.35</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">primary, CR1 by year</td>
<td style="text-align: right;"><span class="math inline">+0.1396</span></td>
<td style="text-align: right;"><span class="math inline">0.0313</span></td>
<td style="text-align: right;"><span class="math inline">4.46</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">primary, quarter-of-year dummies</td>
<td style="text-align: right;"><span class="math inline">+0.1396</span></td>
<td style="text-align: right;"><span class="math inline">0.0296</span></td>
<td style="text-align: right;"><span class="math inline">4.72</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">no-retail control (XOM, T, VZ, UNP)</td>
<td style="text-align: right;"><span class="math inline">+0.1453</span></td>
<td style="text-align: right;"><span class="math inline">0.0361</span></td>
<td style="text-align: right;"><span class="math inline">4.03</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">degree 2, bw 3, donut</td>
<td style="text-align: right;"><span class="math inline">+0.3022</span></td>
<td style="text-align: right;"><span class="math inline">0.0914</span></td>
<td style="text-align: right;"><span class="math inline">3.30</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: placebo cutoffs</em></td>
</tr>
<tr>
<td style="text-align: left;">BIS export controls, <span class="math inline"><em>t</em><sub>0</sub> = 2022.767</span>, bw 3</td>
<td style="text-align: right;"><span class="math inline">+0.1253</span></td>
<td style="text-align: right;"><span class="math inline">0.0317</span></td>
<td style="text-align: right;"><span class="math inline">3.95</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2021.5</span> (no event), full sample, bw 2</td>
<td style="text-align: right;"><span class="math inline">−0.3135</span></td>
<td style="text-align: right;"><span class="math inline">0.0452</span></td>
<td style="text-align: right;"><span class="math inline">6.93</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2019.75</span> (no event), pre-only, bw 2</td>
<td style="text-align: right;"><span class="math inline">+0.2627</span></td>
<td style="text-align: right;"><span class="math inline">0.0514</span></td>
<td style="text-align: right;"><span class="math inline">5.11</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2019.5</span> (no event), pre-only, bw 2</td>
<td style="text-align: right;"><span class="math inline">+0.2843</span></td>
<td style="text-align: right;"><span class="math inline">0.0736</span></td>
<td style="text-align: right;"><span class="math inline">3.86</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
</tbody>
</table>

Notes: HC1 standard errors against standard normal critical values except where noted; donut $`=0.13`$ yr (one quarter); bandwidths in years. Outcome: $`\log_{10}`$ group capex, \$bn per quarter. The pre-only placebo battery (sample $`t<2022.7`$, 15 cutoffs 2017.0–2020.5, bw 2) rejects at nominal 5% in 7/15 cases; the placebo-calibrated $`p`$-value of the ChatGPT estimate at the matched bandwidth is $`0.125`$ ($`|z|`$) and $`0.250`$ ($`|\tau|`$).

# Caveats and Conclusion

Three caveats are decisive. First, the nominal inference is an artifact: HC1/CR1 standard errors assume serial independence, but the outcome is a pair of smooth, serially correlated aggregates (residual lag-1 autocorrelation up to $`0.66`$), and the placebo battery measures the resulting size distortion directly — a 5% test rejects 47% of the time at non-event dates, the failure mode documented by and exactly what placebo distributions exist to catch . Second, a calendar-time cutoff absorbs every late-2022 shock: the BIS export controls fall eight weeks before ChatGPT and produce a statistically indistinguishable estimate, so the design cannot name the shock. Third, the panel is two aggregated units observed 46 times; with $`\leq12`$ quarters per DiK cell, local slopes are estimated from few points on smooth curves, and the full-sample 2021.5 kink shows that this setting manufactures nominal $`|z|>6`$ where nothing happened.

The verdict is descriptive-only. The post-ChatGPT divergence in capex slopes — big-4 growth roughly doubling while the control aggregate’s flattens — is a genuine feature of the data: sign-robust across every specification, localized in 2023.0–2023.5 exactly as a lagged investment response to a late-2022 shock would be, and visible per firm for three of the four hyperscalers. But the design cannot certify it causally, and its headline nominal $`p\approx4\times10^{-6}`$ overstates the evidence by roughly five orders of magnitude relative to the placebo-calibrated $`p`$ of $`0.125`$–$`0.25`$. Something bent hyperscaler capex around the turn of 2023; these data cannot say it was ChatGPT.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 ; estimation is deterministic (local weighted least squares), and seed 0 was passed wherever the CLI accepts one. The EDGAR pulls are not committed to the repository; the run of record retains the derived panel and all estimate files. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Acemoglu, D. (2025). The simple macroeconomics of AI. *Economic Policy*, 40(121), 13–58.

Allen, G. C. (2022). *Choking Off China’s Access to the Future of AI*. Center for Strategic and International Studies, 11 October 2022. <https://www.csis.org/analysis/choking-chinas-access-future-ai>

Bertrand, M., Duflo, E., and Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *The Quarterly Journal of Economics*, 119(1), 249–275.

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Eisfeldt, A. L., Schubert, G., and Zhang, M. B. (2023). Generative AI and firm values. NBER Working Paper No. 31222.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
