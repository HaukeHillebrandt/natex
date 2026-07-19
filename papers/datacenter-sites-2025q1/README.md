> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/datacenter-sites-2025q1/) · [PDF in this repo](./main.pdf)

# Introduction

January 2025 was the most crowded month in the short history of AI infrastructure policy: the US diffusion rule was published on 13 January, DeepSeek-R1 was released on 20 January , and the \$500-billion Stargate project was announced on 21 January . A natural reading of subsequent buildout data is that 2025Q1 marks a regime change — the quarter the industry shifted from ordinary expansion to a land rush — and that the shift is concentrated in the “neocloud” operators (CoreWeave, xAI/Colossus, the Stargate sites, and peers) whose business is AI compute rather than general cloud. This paper tests both claims at the site level and rejects both, in an instructive way: the industry-wide “break” is real in levels but is not a break, and the neocloud attribution is a clean null.

The buildout itself is well documented. Data centers are the engine rooms of AI: industrial facilities whose construction, power, and cooling determine where frontier compute can exist . The leading AI supercomputers have doubled in performance roughly every nine months, with hardware cost and power needs doubling yearly , and the International Energy Agency projects global data-centre electricity demand to more than double by 2030, driven chiefly by AI . What has been missing is site-level measurement: Epoch AI’s Frontier Data Centers hub now tracks individual campuses from satellite imagery, permits, and disclosures, making the timing question empirically addressable.

Methodologically, the exercise sits where two cautionary literatures meet. Breaks located in *time* lack the cross-sectional randomization logic of standard quasi-experimental designs, and naive inference on a single temporal cutoff is fragile ; when identification comes from one calendar event hitting one group, asymptotic cluster inference is unreliable and permutation-style inference over placebo assignments is the appropriate discipline . We add a third, automation-specific caution: when a discovery algorithm is handed a panel whose treatment coding was *constructed* from the hypothesis under test, an exact recovery of the hypothesized date is confirmation of the coding, not of the outcome. The scout pass that preceded this study made exactly that misreading; the role-assignment triage below corrects it.

# Data

The input is Epoch AI’s Frontier Data Centers hub timelines : 411 dated site snapshots covering 73 sites (2018-12-18 to 2030-01-01), extracted 2026-07-19. The 73 rows dated on or after 2026-07-25 are stated projections and are excluded; Section <a href="#sec:results" data-reference-type="ref" data-reference="sec:results">4</a> shows this filter is provably a no-op for the analysis panel. Each site’s “IT power (MW)” is carried forward (LOCF) as a step function evaluated at quarter ends 2023Q4 to 2026Q2, with a site at 0 MW before its first snapshot; quarterly *additions* are first differences with negative revisions floored at 0; the site-level outcome is $`\operatorname{asinh}(\text{additions}/10)`$. Time is indexed $`t = 4\,\text{year} + (\text{quarter}-1)`$, so $`2025\text{Q1} = 8100`$. The neocloud group comprises 21 sites with hypothesized treatment onset 2025Q1: Colossus 1/2, seven CoreWeave sites, Fluidstack Lake Mariner, eight OpenAI Stargate sites, Oracle Batam, Start Campus Sines, and VNET Bayin Ulanqab. The known-treatment comparison panel codes Colossus 1 alone as treated. Baseline panels are bitwise identical to the scout pass’s.

# Design and Methods

#### Seeded discovery scans.

Two `natex` SuDDDS difference-in-differences scans (Bernoulli model, within-cluster-copula inference, $`q=99`$ permutations, seed 0) search the panels for the best-supported treatment date $`t_0`$ and window $`W`$: one on the neocloud coding, one on the Colossus known-treatment coding. Crucially, SuDDDS discovery reads only the design columns — unit, time, and the treatment indicator $`\theta`$ — and never the outcome. Both panels *construct* $`\theta = \text{group} \times \mathbf{1}[t \geq t_0]`$, so an exact recovery of $`t_0`$ certifies that the constructed assignment is scannable, not that MW additions break there. Outcome evidence must come from the two designs below.

#### Aggregate break-date placebos.

On total tracked additions we compute, at every feasible break date $`c`$ (six dates with at least three quarters per side), three statistics: S1, the post$`-`$pre shift in mean log additions; S2, the slope change (kink) in the MW-level series at $`c`$; and S3, the kink in *log* capacity at $`c`$, i.e. growth acceleration. S2 and S3 are single-series analogues of the kink contrasts formalized in the difference-in-kinks literature and implemented in `natex` . If 2025Q1 is a genuine regime break it should be the argmax of the grid; its rank across placebo dates is the test, in the spirit of randomization inference over assignments , sized for a coarse grid (placebo-$`p`$ floor $`1/6`$).

#### Neocloud attribution.

A two-way fixed-effects (TWFE) regression of $`\operatorname{asinh}(\text{additions}/10)`$ on $`\text{neocloud} \times \mathbf{1}[t \geq 2025\text{Q1}]`$ with site and quarter fixed effects (73 sites $`\times`$ 11 quarters) estimates the neocloud-specific post-2025Q1 shift $`\tau`$. Because the contrast is a single calendar event hitting one group, inference is by label permutation : 199 draws (seed 0) reassigning the 21-site label among all 73 sites, plus a chip-protocol variant reassigning among the 52 controls. CR1 cluster-by-site standard errors are reported alongside and agree.

#### Decomposition and sensitivity.

Post-2025Q1 additions are decomposed by site and group. Sensitivity variants truncate the panel end at 2026Q1 and 2025Q4, and an incumbent-only variant keeps the 59 sites first tracked on or before 2024-12-31 (12 treated) to separate real expansion from satellite-coverage entry .

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Tracked quarterly IT-power additions, stacked by group: the post-2025Q1 rise is roughly fivefold and occurs in <em>both</em> groups; the largest single quarters are 2025Q4 and 2026Q2, not 2025Q1. (b) Log total tracked capacity is near-linear (<span class="math inline">∼</span>0.4 log points per quarter) with mildly decelerating growth — the levels “break” is smooth exponential growth read in levels. (c) Placebo break-date grid for S1, the post<span class="math inline">−</span>pre shift in mean log additions: 2025Q1 (vermilion) ranks 3 of 6; the argmax is 2025Q4. (d) Top-10 site contributions to post-2025Q1 additions: broad-based and hyperscaler-led, with the neocloud group (orange) carrying 23.7% overall.</figcaption>
</figure>

#### The surge is real in levels.

Tracked additions average 296.8 MW/q over 2023Q4–2024Q4 and 1,516.9 MW/q over 2025Q1–2026Q2 — a ratio of 5.11 — with complete separation: the smallest post-2025Q1 quarter (537 MW) exceeds the largest pre-quarter (448 MW). Nothing below disputes this; the question is whether it is a *localized break at 2025Q1*.

#### But 2025Q1 is never the best break date.

On the six-date placebo grid (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, Panel A), 2025Q1 is the argmax of none of the three statistics. The S1 log-additions shift at 2025Q1 is 1.488, rank 3 of 6 (placebo $`p=0.50`$); the argmax is 2025Q4 at 1.665. The S2 level kink at 2025Q1 is 1,354 MW/q, dead last — rank 6 of 6 ($`p=1.00`$), argmax 2025Q4 at 2,119. The S3 log-level kink is $`-0.100`$ at 2025Q1 and *negative at all six candidate dates*: quarterly log growth of tracked capacity decelerates from 0.404 to 0.327 log/q. A series growing smoothly at $`\sim`$<!-- -->40% per quarter mechanically makes every later quarter’s additions exceed every earlier one’s; the “break” is the exponential, not a regime change. The largest single jumps are 2025Q4 ($`+1{,}936`$ MW) and 2026Q2 ($`+3{,}610`$ MW). Across all sensitivity variants, 2025Q1 is the argmax in 1 of 12 variant-statistic cells (the 2025Q4-truncated S3, placebo $`p=0.25`$ on a four-date grid).

#### The SuDDDS 2025Q1 hit is mechanical.

The seeded neocloud scan returns max LLR $`=24.114`$, $`t_0 = 8100 = 2025`$Q1 *exactly*, $`W=3`$, scan $`p=0.010`$ (the $`q=99`$ permutation floor), and passes composition validation. This is the scout pass’s headline — and it is a recovery of the constructed treatment coding, not outcome evidence: discovery never reads the outcome (Section <a href="#sec:methods" data-reference-type="ref" data-reference="sec:methods">3</a>), and the scan’s in-scan effect estimators refused in both runs (0 usable placebo units of the required 5), so the scans contribute no outcome estimate. The Colossus known-treatment scan illustrates the power ceiling with 1 treated site of 73: LLR $`=2.537`$, $`t_0=2024`$Q3, $`W=5`$, scan $`p=0.230`$, deterministic on re-run (the scout logged $`p=0.28`$ from a marginally earlier code state with identical LLR/$`t_0`$/$`W`$).

#### Neocloud attribution is a clean null.

The TWFE contrast gives $`\tau = +0.0976`$ asinh-MW (CR1 SE $`=0.1396`$, $`t=0.699`$). The 199-draw label permutation puts the observed $`\tau`$ well inside the null: $`p=0.560`$ relabeling among all 73 sites, $`p=0.555`$ under the chip protocol, with a permutation $`\tau`$ SD of 0.150 — the estimate is $`\sim`$<!-- -->0.65 null SDs from zero. Truncating the panel at 2026Q1 or 2025Q4 leaves the null intact ($`p=0.605`$, $`p=0.505`$). The incumbent-only variant is larger but still not significant: $`\tau=+0.258`$, $`t=1.49`$, permutation $`p=0.155`$.

#### Broad-based and hyperscaler-led.

Post-2025Q1 additions total 9,101 MW, of which the 21-site neocloud group carries 2,158 MW — 23.7%. Concentration is moderate: the top 5 sites carry 33.9% and the top 10 carry 49.1%. The leaders are the Anthropic–Amazon New Carlisle campus at 910 MW (10.0%), Microsoft Fairwater Atlanta (636), Meta Prometheus (631), Colossus 2 (490, neocloud), and OpenAI Stargate Abilene (421, neocloud). Inside the scan window 2025Q1–2025Q3 (2,245 MW) the largest contributors are New Carlisle (398 MW, 17.7%) and Amazon Madison (284 MW, 12.7%) — hyperscaler campuses, not neoclouds.

#### Sensitivity and the projection filter.

Excluding stated projections is provably a no-op: all 73 projection rows are dated on or after 2026-07-25, later than the last quarter-end 2026-06-30, and every site has a real snapshot on or before 2026-06-30, so the 2023Q4–2026Q2 panel is bitwise invariant to any snapshot-filter date from 2026-06-30 on (including no filter). The incumbent variant shows the surge is not satellite-coverage entry: the 59 sites first tracked by 2024-12-31 carry 3,074 of the 3,610 MW of 2026Q2 additions ($`\sim`$<!-- -->85%).

<table id="tab:main">
<caption>Headline estimates. Panel A: aggregate break-date placebo grid (six feasible dates, <span class="math inline"> ≥ 3</span> quarters per side); the test is 2025Q1’s rank. Panel B: seeded <code>natex</code> SuDDDS scans (Bernoulli/WCC, <span class="math inline"><em>q</em> = 99</span>, seed 0). Panel C: TWFE neocloud attribution with 199-draw label-permutation inference (seed 0).</caption>
<thead>
<tr>
<th style="text-align: left;">Quantity</th>
<th style="text-align: right;">Estimate</th>
<th style="text-align: right;">Rank / <span class="math inline"><em>t</em></span></th>
<th style="text-align: right;"><span class="math inline"><em>p</em></span></th>
<th style="text-align: left;">notes</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: is 2025Q1 the best break date? (placebo grid, <span class="math inline"><em>n</em> = 6</span> dates)</em></td>
</tr>
<tr>
<td style="text-align: left;">S1 shift in mean log additions</td>
<td style="text-align: right;"><span class="math inline">1.488</span></td>
<td style="text-align: right;">3/6</td>
<td style="text-align: right;"><span class="math inline">0.50</span></td>
<td style="text-align: left;">argmax 2025Q4 (<span class="math inline">1.665</span>)</td>
</tr>
<tr>
<td style="text-align: left;">S2 level kink (MW/q)</td>
<td style="text-align: right;"><span class="math inline">1, 354</span></td>
<td style="text-align: right;">6/6</td>
<td style="text-align: right;"><span class="math inline">1.00</span></td>
<td style="text-align: left;">argmax 2025Q4 (<span class="math inline">2, 119</span>)</td>
</tr>
<tr>
<td style="text-align: left;">S3 log-level kink (log/q)</td>
<td style="text-align: right;"><span class="math inline">−0.100</span></td>
<td style="text-align: right;">4/6</td>
<td style="text-align: right;"><span class="math inline">0.67</span></td>
<td style="text-align: left;">negative at all 6 dates</td>
</tr>
<tr>
<td style="text-align: left;">post/pre mean additions ratio</td>
<td style="text-align: right;"><span class="math inline">5.11</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">min post <span class="math inline">537&gt;</span> max pre <span class="math inline">448</span></td>
</tr>
<tr>
<td style="text-align: left;">log growth, pre <span class="math inline">→</span> post</td>
<td style="text-align: right;"><span class="math inline">0.404 → 0.327</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">deceleration</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: SuDDDS scans (discovery reads <span class="math inline">(<em>x</em>, <em>t</em>, <em>θ</em>)</span> only — never the outcome)</em></td>
</tr>
<tr>
<td style="text-align: left;">neocloud scan, max LLR</td>
<td style="text-align: right;"><span class="math inline">24.114</span></td>
<td style="text-align: right;"><span class="math inline"><em>t</em><sub>0</sub> = 2025</span>Q1, <span class="math inline"><em>W</em> = 3</span></td>
<td style="text-align: right;"><span class="math inline">0.010</span></td>
<td style="text-align: left;">recovers the coding</td>
</tr>
<tr>
<td style="text-align: left;">Colossus scan, max LLR</td>
<td style="text-align: right;"><span class="math inline">2.537</span></td>
<td style="text-align: right;"><span class="math inline"><em>t</em><sub>0</sub> = 2024</span>Q3, <span class="math inline"><em>W</em> = 5</span></td>
<td style="text-align: right;"><span class="math inline">0.230</span></td>
<td style="text-align: left;">1 treated of 73</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: neocloud attribution, TWFE <span class="math inline"><em>τ</em></span> (asinh-MW), site+quarter FE</em></td>
</tr>
<tr>
<td style="text-align: left;">baseline (21/73 treated)</td>
<td style="text-align: right;"><span class="math inline">+0.098</span></td>
<td style="text-align: right;"><span class="math inline"><em>t</em> = 0.699</span></td>
<td style="text-align: right;"><span class="math inline">0.560</span></td>
<td style="text-align: left;"><span class="math inline"><em>τ</em></span> SD <span class="math inline">0.150</span>; chip <span class="math inline"><em>p</em> = 0.555</span></td>
</tr>
<tr>
<td style="text-align: left;">truncate 2026Q1 / 2025Q4</td>
<td style="text-align: right;"><span class="math inline">0.088/0.112</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.605/0.505</span></td>
<td style="text-align: left;">conclusions unchanged</td>
</tr>
<tr>
<td style="text-align: left;">incumbent-only (12/59 treated)</td>
<td style="text-align: right;"><span class="math inline">+0.258</span></td>
<td style="text-align: right;"><span class="math inline"><em>t</em> = 1.49</span></td>
<td style="text-align: right;"><span class="math inline">0.155</span></td>
<td style="text-align: left;">larger, still n.s.</td>
</tr>
</tbody>
</table>

Notes: Panel A placebo $`p`$ is the rank of 2025Q1 divided by the grid size (floor $`1/6`$). Panel B scan $`p`$-values are $`q=99`$ within-cluster-copula permutation floors; the neocloud scan’s exact $`t_0`$ recovery certifies the constructed 21/73-site assignment is scannable, not that the outcome breaks there (in-scan effect estimators refused: 0 usable placebos of $`\geq 5`$ required). Panel C $`p`$-values are 199-draw label permutations, seed 0; CR1 cluster-by-site $`t`$-statistics agree with the permutation verdicts.

# Caveats and Conclusion

Six caveats bound the claims. First, *LOCF timing*: snapshots are sparse (median 4 per site), so carried-forward power smears true addition dates by up to one inter-snapshot gap; quarter-level timing is approximate, and non-monotone estimate revisions are floored at 0. Second, *tracked-universe growth*: the hub’s coverage grows from 4 to 54 sites with nonzero power over the panel, so aggregate levels lower-bound industry capacity and early growth rates are inflated by coverage catch-up — a bias *toward* finding late acceleration, which strengthens the no-2025Q1-break reading, though absolute MW shares remain conditional on Epoch’s tracking choices . Third, *coarse placebo grids*: with 4–6 feasible dates the placebo $`p`$ has a floor of $`1/6`$–$`1/4`$; statements are about rank, not tight $`p`$-values. Fourth, a *crowded quarter*: Stargate (21 January), DeepSeek-R1 (20 January), and the diffusion rule (13 January) all land in 2025Q1 , so even a real 2025Q1 break could not separate these candidate causes. Fifth, *calendar-time inference*: the neocloud contrast is one calendar event hitting one group; we therefore rest on permutation-in-space inference , and the asymptotic CR1 statistics happen to agree. Sixth, *scale*: $`\operatorname{asinh}(\text{additions}/10)`$ compresses large sites, and $`\tau`$ lives on that scale.

The conclusion is a double negative with positive content. The post-2025Q1 surge in tracked datacenter additions is real, large, and completely separated in levels — but it is what smooth $`\sim`$<!-- -->40%-per-quarter exponential growth of tracked capacity looks like, with 2025Q1 never the best break date on any statistic, growth mildly decelerating, and the largest jumps arriving in 2025Q4 and 2026Q2. The one seemingly sharp result — a seeded discovery scan returning 2025Q1 exactly at $`p=0.010`$ — is a mechanical recovery of the constructed treatment coding, a failure mode worth naming for any pipeline that scans panels whose design columns encode the hypothesis. And the group story fails directly: neocloud attribution is a clean permutation null, with the buildout led by hyperscaler campuses — first among them the Anthropic–Amazon New Carlisle site at 10% of all post-2025Q1 additions. The Stargate quarter did not change the growth regime; the regime was already exponential .

#### Reproducibility.

All estimates derive from the public Epoch AI Frontier Data Centers hub timelines ; no dataset files are committed. The only stochastic steps are the seeded scan and permutation draws (seed 0 throughout). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates from the committed `figures/make_fig.py`, which asserts the headline estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Conley, T. G., and Taber, C. R. (2011). Inference with “difference in differences” with a small number of policy changes. *Review of Economics and Statistics*, 93(1), 113–125.

DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948.

Epoch AI (2026). Frontier Data Centers Hub. <https://epoch.ai/data/data-centers> (timelines extracted 19 July 2026).

Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation*. Software, version 0.2.0. <https://github.com/HaukeHillebrandt/natex>

International Energy Agency (2025). *Energy and AI*. IEA, Paris. <https://www.iea.org/reports/energy-and-ai>

OpenAI (2025). Announcing The Stargate Project. Company announcement, 21 January 2025. <https://openai.com/index/announcing-the-stargate-project/>

Pilz, K., and Heim, L. (2023). Compute at scale: A broad investigation into the data center industry. arXiv:2311.02651.

Pilz, K. F., Sanders, J., Rahman, R., and Heim, L. (2025). Trends in AI supercomputers. arXiv:2504.16026.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
