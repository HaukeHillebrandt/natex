> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/btos-spliced-r1-extension/) · [PDF in this repo](./main.pdf)

# Introduction

When an event-study estimate is nominally significant but its falsification battery is ambiguous, the cleanest arbiter is data that did not exist when the estimate was made. This paper supplies that arbiter for one specific claim. A companion analysis of the Census Bureau’s Business Trends and Outlook Survey (BTOS) through October 2025 found that measured AI use in AI-exposed sectors accelerated sharply at exactly the first survey wave after the release of the DeepSeek-R1 reasoning model on 20 January 2025 : a data-driven scan localized the break at that wave, and a group difference-in-kinks (DiK) estimate at the release date was $`+7.3`$ pp/yr (cluster-robust se $`2.8`$). That paper already graded the finding descriptive-only, because placebo cutoffs at non-event 2024 dates produced slope breaks as large as the R1 estimate. Nine more months of BTOS data now exist. They support a sharper test: if a genuine R1 response bent the adoption curve, the kink should persist — and its precision improve — as the estimation window widens to include the new data; if the “bend” was local curvature in a steep diffusion path, it should dissolve.

Running that test requires carrying the BTOS series across a measurement break. The BTOS has asked US firms since September 2023 whether they used AI in the last two weeks ; on 3 December 2025 the Census Bureau announced that the AI questions had been reworded — from AI use “in producing goods or services” to AI use “in any of its business functions” — and started a new time series . A second companion paper estimates the rewording jump in a known-cutoff regression-discontinuity-in-time design and calibrates a splice factor: the new-wording national level is 1.553 times the old-wording level at the cutoff. This paper divides new-wording values by that ratio (with per-sector and covariate-adjusted variants as robustness), which is what makes the extended panel usable at all.

Methodologically the paper applies the DiK design of , which builds on the regression kink design of , using the `natex` toolkit . Kink estimates with calendar time as the running variable inherit the known pathologies of regression discontinuity in time , and slope breaks in time series are notoriously easy to “find,” which is why placebo estimates at non-event dates — the permutation logic of — are the decision criterion throughout, rather than any single cluster-robust $`p`$-value resting on 6–7 sector clusters . The result is a clean reversal: the prior estimate replicates exactly on its own window, every extended-window specification is null to negative, and the placebo landscape now brackets the R1 date with significant breaks of *both* signs at dates where nothing happened.

# Data

Two BTOS vintages are combined. The old-wording segment is the frozen AI core workbook of the companion paper (DRB approval CBDRB-FY25-0425): sector-level biweekly estimates and standard errors for Question 7 (“In the last two weeks, did this business use Artificial Intelligence (AI) in producing goods or services?”), 54 waves 202319–202520, collection periods 2023-09-11 through 2025-10-05. The new-wording segment is a fresh census.gov sector download (fetched 2026-07-19): the reworded question (“…in any of its business functions?”), 17 waves 202524–202614, December 2025 through June 2026. Waves 202521–202523 were never collected (the October–November 2025 federal government shutdown), leaving a 7-week no-data gap at $`t \in [2025.78, 2025.92]`$ immediately before the rewording. The running variable $`t`$ is the fractional year of each wave’s collection-period midpoint.

The spliced panel divides new-wording values by 1.553, the national new/old measurement-RDD ratio at the 2025-12-03 rewording , and contains 1,121 sector-wave cells across 19 NAICS sectors, $`t = 2023.710`$–$`2026.507`$, of which 294 cells are new-wording. The exposed group is {51 information, 52 finance and insurance, 54 professional/scientific/technical services}; the comparison group is {11, 21, 23, 72}, thin as in the companion paper: sectors 11 and 21 have only $`\sim`$<!-- -->3 usable old-wording waves each, so the unexposed side is effectively construction (23) and accommodation and food services (72) before 2025. Two robustness panels vary the measurement handling: (i) per-sector splice factors estimated at the rewording boundary (old-side local linear trend extrapolated to the first new-wording waves; factors range 1.19–4.08, larger in low-adoption sectors, national fallback where fewer than two boundary observations exist); and (ii) the raw unspliced outcome with an additive new-wording dummy as a covariate.

# Design and Methods

#### Group difference-in-kinks.

Following , the DiK estimand is the change in the slope of the outcome in calendar time at a known cutoff for the exposed group minus the same slope change for the comparison group, estimated by local linear regression on each side of the cutoff within each group (`natex` v0.2.0 ). All specifications use a triangular kernel, a one-wave donut (0.0384 yr), and CR1 standard errors clustered by sector, with $`p`$-values from $`t(G-1)`$ with $`G \in
\{6,7\}`$ clusters . The R1 cutoff is $`t_0=2025.055`$ (2025-01-20; first post wave 202503). Bandwidth 0.5 yr reproduces the companion paper’s primary specification — its window $`[2024.56, 2025.56]`$ ends in July 2025 and therefore contains *zero* post-extension data — while bandwidths 1.0 and 1.5 yr are the out-of-sample test: their windows extend to January and July 2026 respectively and see the new waves.

#### Falsification battery.

The same estimator runs at non-event placebo cutoffs (2024.30, 2024.50, 2024.85, 2025.30, 2025.60), at the splice boundary itself (2025.845, a pure measurement date), and at OpenAI’s o1 release (2024-09-12, $`t=2024.6967`$) — 36 deterministic `natex` kink runs in total across the three measurement panels (the kink CLI has no RNG). In the spirit of , the placebo grid, not any single $`p`$-value, is the decision criterion: with calendar time as the running variable, any macro or sector-specific shock aliases as a kink , so an R1 effect must stand out *against* the placebo landscape, not merely reject zero.

#### Measurement handling, stated in advance.

A national splice ratio applied to all sectors is wrong in detail whenever the rewording effect is sector-heterogeneous — and it is (boundary factors 1.19–4.08). The per-sector-splice panel addresses this directly. The covariate variant (raw outcome plus an additive new-wording dummy) is included for completeness but is structurally misspecified: the rewording is a multiplicative rescale ($`\sim`$<!-- -->1.55$`\times`$ nationally), and a single additive dummy common to both groups cannot absorb a shift that lifts the exposed group ($`\sim`$<!-- -->24 pp raw) far more than the unexposed ($`\sim`$<!-- -->5 pp). The symptom to watch for is the covariate spec “finding” significant kinks at placebo cutoffs; it does (Section <a href="#sec:results" data-reference-type="ref" data-reference="sec:results">4</a>), so its R1 estimates are discounted.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Spliced BTOS AI-use share by exposure group across all 71 waves (new-wording waves divided by 1.553); vertical lines mark o1, DeepSeek-R1, and the rewording splice (dotted), with the shaded band the 7-week shutdown no-data gap. (b) Group DiK estimates with 95% CR1 confidence intervals at the R1 cutoff (vermillion: the bandwidth-0.5 replication, whose window ends in July 2025, and the extended bandwidth-1.0/1.5 estimates, which see the new data) and at placebo cutoffs (gray, bandwidth 1.0); filled markers are significant at the 5% level. The extended R1 estimates are null while non-event cutoffs on both sides are significantly positive (2024) and negative (2025H2).</figcaption>
</figure>

#### The replication is exact; the extension is null.

At bandwidth 0.5 the R1 DiK is $`\hat\tau = +7.27`$ pp/yr (se $`2.81`$, 95% CI $`[0.38, 14.16]`$, $`p=0.042`$, $`n=123`$) — to the digit the companion paper’s primary estimate, as it must be: the spliced panel is byte-identical to the old panel on that window. The moment the window widens to include any of the nine new months, the estimate collapses: bandwidth 1.0 gives $`\hat\tau = -0.16`$ pp/yr (se $`3.44`$, 95% CI $`[-8.57, +8.24]`$, $`p=0.963`$, $`n=244`$, 7 clusters) and bandwidth 1.5 gives $`-2.09`$ (se $`3.24`$, 95% CI $`[-10.03, +5.85]`$, $`p=0.543`$, $`n=373`$). This is not bandwidth dilution of a real kink — a persistent slope change measured with more data on both sides would gain, not lose, precision-weighted magnitude; it is the signature of local curvature that a short window mistook for a bend.

#### Robust to how the splice is done.

With per-sector splice factors the extended-window R1 estimates are $`+1.93`$ (se $`2.82`$, $`p=0.520`$) at bandwidth 1.0 and $`+2.05`$ (se $`1.90`$, $`p=0.324`$) at bandwidth 1.5 — positive but small and nowhere near significant. The covariate variant gives $`+4.73`$ (se $`3.15`$, $`p=0.185`$) at bandwidth 1.0 and $`+8.04`$ (se $`2.64`$, $`p=0.023`$) at bandwidth 1.5 — the only significant extended R1 estimate in the grid, and it is discounted for the reason stated in advance: the same specification “finds” significant positive kinks at non-event placebo cutoffs ($`+8.52`$, $`p=0.014`$ at $`t_0=2025.30`$; $`+5.35`$, $`p=0.009`$ at $`t_0=2025.60`$), because an additive dummy cannot absorb a multiplicative $`\sim`$<!-- -->1.55$`\times`$ measurement rescale. A specification that fails its own placebo battery cannot rescue the effect it was built to test.

#### The placebo landscape is now two-sided.

On the extended national-splice panel, non-event cutoffs in 2024 still produce positive DiKs as large as the old R1 estimate — $`+11.50`$ (se $`3.58`$, $`p=0.018`$) at $`t_0=2024.30`$ bandwidth 0.5, $`+4.23`$ ($`p=0.003`$) at bandwidth 1.0; $`+3.98`$ (se $`0.90`$, $`p=0.007`$) and $`+3.57`$ ($`p=0.002`$) at $`t_0=2024.50`$ — and non-event cutoffs in 2025H2 produce significant *negative* DiKs: $`-11.77`$ ($`p=0.007`$) at $`t_0=2025.30`$ bandwidth 0.25, $`-16.70`$ ($`p=0.019`$) and $`-9.08`$ (se $`2.49`$, $`p=0.011`$) at $`t_0=2025.60`$, and $`-4.06`$ ($`p=0.019`$) at the splice boundary itself (Table <a href="#tab:dik" data-reference-type="ref" data-reference="tab:dik">1</a>; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). A gap curve that is convex through 2024 and plateauing from mid-2025 generates exactly this pattern at *any* cutoff; R1 is nowhere special on it. The o1 contrast remains bandwidth-aliased, as in the companion paper: $`-5.91`$ ($`p=0.099`$) at bandwidth 0.5, $`+3.82`$ ($`p=0.036`$) at 1.0, $`+2.73`$ ($`p=0.192`$) at 1.5.

#### Descriptively, the gap decelerated.

The SE-weighted exposed-minus-unexposed gap slope is $`+4.38`$ pp/yr in the year *before* R1 and $`+2.27`$ pp/yr in the year *after* — a deceleration, not an acceleration. The spliced gap rises from 8.0 pp at the panel start (September 2023) to roughly 18 pp by mid-2026 and is approximately flat over February–June 2026 (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>a). The companion paper’s “descriptive acceleration” — a gap slope tripling measured over $`\pm`$<!-- -->13 waves — was a short-window artifact of this convex-then-plateauing path.

<table id="tab:dik">
<caption>Group difference-in-kinks estimates on the extended panel, BTOS AI-use share (pp/yr).</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline"><em>p</em></span></th>
<th style="text-align: center;">sig. 5%</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: DeepSeek-R1 cutoff (<span class="math inline"><em>t</em><sub>0</sub> = 2025.055</span>), national splice</em></td>
</tr>
<tr>
<td style="text-align: left;">bw 0.5 (replication; window ends 2025.56)</td>
<td style="text-align: right;"><span class="math inline">+7.27</span></td>
<td style="text-align: right;"><span class="math inline">2.81</span></td>
<td style="text-align: right;"><span class="math inline">0.042</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">bw 1.0 (extended)</td>
<td style="text-align: right;"><span class="math inline">−0.16</span></td>
<td style="text-align: right;"><span class="math inline">3.44</span></td>
<td style="text-align: right;"><span class="math inline">0.963</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">bw 1.5 (extended)</td>
<td style="text-align: right;"><span class="math inline">−2.09</span></td>
<td style="text-align: right;"><span class="math inline">3.24</span></td>
<td style="text-align: right;"><span class="math inline">0.543</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: R1 cutoff, alternative measurement handling</em></td>
</tr>
<tr>
<td style="text-align: left;">per-sector splice, bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+1.93</span></td>
<td style="text-align: right;"><span class="math inline">2.82</span></td>
<td style="text-align: right;"><span class="math inline">0.520</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">per-sector splice, bw 1.5</td>
<td style="text-align: right;"><span class="math inline">+2.05</span></td>
<td style="text-align: right;"><span class="math inline">1.90</span></td>
<td style="text-align: right;"><span class="math inline">0.324</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">raw <span class="math inline">+</span> wording dummy, bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+4.73</span></td>
<td style="text-align: right;"><span class="math inline">3.15</span></td>
<td style="text-align: right;"><span class="math inline">0.185</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">raw <span class="math inline">+</span> wording dummy, bw 1.5</td>
<td style="text-align: right;"><span class="math inline">+8.04</span></td>
<td style="text-align: right;"><span class="math inline">2.64</span></td>
<td style="text-align: right;"><span class="math inline">0.023</span></td>
<td style="text-align: center;">(<span class="math inline">*</span>)</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: placebo and secondary cutoffs, national splice</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.30</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">+11.50</span></td>
<td style="text-align: right;"><span class="math inline">3.58</span></td>
<td style="text-align: right;"><span class="math inline">0.018</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.30</span> (no event), bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+4.23</span></td>
<td style="text-align: right;"><span class="math inline">0.87</span></td>
<td style="text-align: right;"><span class="math inline">0.003</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.50</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">+3.98</span></td>
<td style="text-align: right;"><span class="math inline">0.90</span></td>
<td style="text-align: right;"><span class="math inline">0.007</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2024.50</span> (no event), bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+3.57</span></td>
<td style="text-align: right;"><span class="math inline">0.65</span></td>
<td style="text-align: right;"><span class="math inline">0.002</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2025.30</span> (no event), bw 0.25</td>
<td style="text-align: right;"><span class="math inline">−11.77</span></td>
<td style="text-align: right;"><span class="math inline">2.68</span></td>
<td style="text-align: right;"><span class="math inline">0.007</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2025.60</span> (no event), bw 0.5</td>
<td style="text-align: right;"><span class="math inline">−16.70</span></td>
<td style="text-align: right;"><span class="math inline">5.27</span></td>
<td style="text-align: right;"><span class="math inline">0.019</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>t</em><sub>0</sub> = 2025.60</span> (no event), bw 1.0</td>
<td style="text-align: right;"><span class="math inline">−9.08</span></td>
<td style="text-align: right;"><span class="math inline">2.49</span></td>
<td style="text-align: right;"><span class="math inline">0.011</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">splice boundary <span class="math inline"><em>t</em><sub>0</sub> = 2025.845</span>, bw 1.0</td>
<td style="text-align: right;"><span class="math inline">−4.06</span></td>
<td style="text-align: right;"><span class="math inline">1.28</span></td>
<td style="text-align: right;"><span class="math inline">0.019</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">o1, <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span>, bw 0.5</td>
<td style="text-align: right;"><span class="math inline">−5.91</span></td>
<td style="text-align: right;"><span class="math inline">2.92</span></td>
<td style="text-align: right;"><span class="math inline">0.099</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;">o1, <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span>, bw 1.0</td>
<td style="text-align: right;"><span class="math inline">+3.82</span></td>
<td style="text-align: right;"><span class="math inline">1.42</span></td>
<td style="text-align: right;"><span class="math inline">0.036</span></td>
<td style="text-align: center;"><span class="math inline">*</span></td>
</tr>
<tr>
<td style="text-align: left;">o1, <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span>, bw 1.5</td>
<td style="text-align: right;"><span class="math inline">+2.73</span></td>
<td style="text-align: right;"><span class="math inline">1.85</span></td>
<td style="text-align: right;"><span class="math inline">0.192</span></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

Notes: CR1 standard errors clustered by sector; $`p`$-values against $`t(G-1)`$ with $`G \in \{6,7\}`$ clusters (5–7 at bandwidth 0.25). Triangular kernel, one-wave donut, bandwidths in years. New-wording waves divided by 1.553 (national splice), by boundary-estimated per-sector factors (Panel B rows 1–2), or entered raw with an additive new-wording dummy (Panel B rows 3–4; discounted — see text: this specification also finds $`+8.52`$, $`p=0.014`$ and $`+5.35`$, $`p=0.009`$ at the 2025.30 and 2025.60 placebo cutoffs). Outcome: sector-level share answering Yes to the BTOS AI-use question, percentage points.

# Caveats and Conclusion

Five caveats bound the reversal, none of which rescues R1. First, the national splice ratio applied to every sector is wrong in detail — the boundary-estimated per-sector factors range from 1.19 to 4.08, larger in low-adoption sectors — but the R1 conclusions are robust to using those factors, which are themselves noisy boundary extrapolations across the 7-week shutdown gap. Second, the covariate variant is structurally misspecified (additive dummy against a multiplicative shift); its lone significant R1 estimate fails its own placebos and should not be cited as evidence. Third, CR1 inference with 6–7 clusters is fragile — but the headline here is a null, and the significant placebos argue *against* the effect, so any over-rejection makes the case against R1 stronger, not weaker. Fourth, the significant negative kink at the splice boundary mixes a genuine gap plateau with possible splice fragility; slopes after $`t=2025.9`$ should be treated as measurement-provisional either way. Fifth, BTOS measures extensive-margin adoption (“used AI in the last two weeks”); a null here does not rule out intensive-margin or within-firm R1 effects that an any-use share cannot see.

The verdict is an honest reversal, and a clean one. The companion paper’s R1 kink replicates exactly on its own window — nothing about the old computation was wrong — and fails completely out of sample: every specification that sees the nine new months of data is null to negative at R1, under two independent splice treatments; the placebo battery now brackets the R1 date with significant breaks of both signs at non-event dates; and the descriptive gap slope decelerated across the release. What looked like an adoption response to DeepSeek-R1 was the local curvature of a diffusion gap that grew convexly through 2024 and plateaued in late 2025 — a pattern that manufactures “significant” kinks at almost any cutoff a researcher might try. The companion paper withheld causal attribution; the extended panel withdraws even the descriptive acceleration.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 from the frozen BTOS workbooks (published at <https://www.census.gov/hfp/btos>; not committed to the repository); the 36 kink runs are fully deterministic (the CLI has no RNG), and the spliced kink panel is byte-identical to the scout build of record. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bonney, K., Breaux, C., Buffington, C., Dinlersoz, E., Foster, L., Goldschlag, N., Haltiwanger, J., Kroff, Z., and Savage, K. (2024). Tracking firm use of AI in real time: A snapshot from the Business Trends and Outlook Survey. NBER Working Paper No. 32319.

Cameron, A. C., and Miller, D. L. (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 645(8081), 633–638.

Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552.

Hillebrandt, H. (2026). A sharp bend, no verdict: Difference-in-kinks tests of US business AI adoption at DeepSeek-R1. Companion paper, natex paper collection. <https://haukehillebrandt.github.io/natex/btos-sector-did-r1/>

Hillebrandt, H. (2026). The question is the treatment: A known-cutoff regression discontinuity at the BTOS AI-question rewording. Companion paper, natex paper collection. <https://haukehillebrandt.github.io/natex/btos-rewording-rdd/>

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

U.S. Census Bureau (2025). *Business Trends and Outlook Survey: AI core question updates*. 3 December 2025. <https://www.census.gov/hfp/btos/downloads/AI%20Question%20Wording%20Updates.pdf>

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
