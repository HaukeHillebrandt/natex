> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/euact-bunching-writeup/) · [PDF](https://haukehillebrandt.github.io/natex/euact-bunching-writeup.pdf) · [PDF in this repo](./main.pdf)

# Introduction

Article 51(2) of the EU AI Act presumes that a general-purpose AI model poses “systemic risk” when the cumulative compute used for its training exceeds $`10^{25}`$ floating-point operations, attaching notification, evaluation, and risk-mitigation obligations to models above the line. The Act entered into force on 1 August 2024 and its general-purpose AI obligations became applicable on 2 August 2025. A bright-line quantity threshold of this kind is, in the language of public economics, a *notch*: crossing it discretely changes the regulatory burden, so agents with values just above the cutoff have an incentive to locate just below it.

A large empirical literature shows that agents do exactly this. developed bunching estimation at kinks in the US income-tax schedule; extended it to notches, showing that a discrete jump in liability produces excess mass just below the cutoff and *missing mass* just above it; surveys the method and its spread beyond taxation to regulation and private-sector prices. The econometric diagnostic for such sorting at a cutoff is the density discontinuity test of : if agents manipulate the running variable, its density jumps at the threshold.

Whether training compute is a good handle for AI governance is actively debated. argue that compute is uniquely governable — detectable, quantifiable, and produced by a concentrated supply chain — and defend training-compute thresholds, including the EU’s $`10^{25}`$-FLOP line, as the most workable trigger for regulatory oversight. counters that compute thresholds are shortsighted and invite exactly the strategic responses studied in the bunching literature. On the empirical side, project model counts above the EU threshold *forward* from pre-Act scaling trends, forecasting 103–306 models above $`10^{25}`$ FLOP by 2028 — a superlinear increase. To our knowledge, no published estimate tests whether the observed post-Act density of models near the line already departs from that trend. This paper supplies that test: a McCrary-style density discontinuity estimate at the statutory cutoff, before versus after the Act’s entry into force, with threshold, date, and timing placebos.

# Data

The input panel is built from Epoch AI’s *Data on AI Models* database , keeping every model with a non-null positive training-compute estimate and a release date from 2 January 2023 to 4 June 2026: 719 models with columns model, date, and $`\log_{10}`$ training FLOP. Event dates are the Act’s entry into force (2024-08-01) and the applicability of general-purpose AI obligations (2025-08-02). The pre-Act period is 2023-01 to 2024-07 ($`n=21`$ models within $`\pm 0.5`$ dex of the cutoff) and the post-Act period is 2024-08 onward ($`n=52`$ in-window). Compute values for recent frontier models are largely Epoch estimates rather than disclosures, and models without any compute estimate ($`\approx 2{,}800`$ of $`\approx 3{,}500`$ database rows) are necessarily excluded — a selection channel we return to in Section <a href="#sec:conclusion" data-reference-type="ref" data-reference="sec:conclusion">5</a>. There is no value heaping near the cutoff: at most 3 duplicate $`\log_{10}`$-compute values occur within $`[24.5, 25.5]`$, so the results below are not an artifact of rounded compute estimates.

# Design and Methods

#### Density discontinuity at the statutory line.

The main estimate is the `natex` toolkit’s McCrary-style binned-Poisson density test, run directly at the statutory cutoff: model counts in equal-width bins of the signed distance $`\log_{10}\mathrm{FLOP}-25.0`$ are fit by a Poisson regression with a linear trend in distance and a jump term at zero; $`\theta`$ is the jump in log density at the cutoff, so $`\theta<0`$ means a deficit just above the line — avoidance bunching below it. The primary specification uses a $`\pm 0.5`$-dex window and 10 bins; sensitivity covers the full grid $`h\in\{0.5,0.75,1.0\}
\times \text{bins}\in\{8,10,12,16,20\}`$ (15 specifications per period). Because the cutoff is fixed by statute rather than searched for, the frozen-geometry caveat that applies to `natex`’s post-discovery density checks does not bite here.

#### Contrasts and placebos.

The pre/post change in bunching is measured two ways: a Fisher exact test on the $`2\times 2`$ table of below/above counts in the $`\pm 0.5`$-dex window by period, and a $`\theta`$-difference (interaction) $`z`$-test between the period-specific density jumps. Specificity is probed with placebo thresholds at $`24.5`$ and $`25.5`$ and a placebo split date of 2023-08-01, which places both “periods” inside the pre-Act sample.

#### Honest inference.

Standard errors for $`\theta`$ are Wald standard errors from the binned Poisson fit and can be optimistic when the underlying density is strongly curved within the window; at the widest bandwidth ($`h=1.0`$) curvature of the steeply falling compute density can leak into $`\theta`$, which is why the narrow-window specification is primary and the wide-window significance is discounted below. All counts, tests, and figures derive deterministically from the input panel: there is no random number generation anywhere in the pipeline.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Model counts in 0.25-dex bins of <span class="math inline">log<sub>10</sub></span> training compute, pre-Act (top) versus post-Act (bottom), colored by side of the statutory <span class="math inline">10<sup>25</sup></span>-FLOP line (dashed). Post-Act, mass accumulates just below the line while the first half-dex above holds 5 models against <span class="math inline">28.9</span> expected at the pre-Act ratio; the frontier jumps far above the notch. (b) Below/above counts per half-year in the <span class="math inline">±0.5</span>-dex window: the above-line count (labeled) collapses from 2025H1, after entry into force and before GPAI applicability, while the just-below count keeps growing.</figcaption>
</figure>

#### A post-Act deficit just above the line.

Post-Act (2024-08-01 onward), the primary density test at $`\log_{10}=25.0`$ gives $`\theta=-1.334`$ (SE $`0.878`$, $`p=0.129`$) with $`n=52`$ in-window models split 47 below / 5 above. Across the full sensitivity grid $`\theta`$ is negative in 15 of 15 specifications (range $`-1.249`$ to $`-2.566`$) and significant at the 5% level in 10 of 15 — all $`h=0.75`$ and $`h=1.0`$ specifications, with significant $`p`$-values from $`0.0010`$ to $`0.0143`$. The pre-Act period shows no such pattern: $`\theta=-0.880`$ (SE $`0.973`$, $`p=0.366`$), $`n=21`$ split 13/8, and 0 of 15 sensitivity specifications reject ($`\theta`$ from $`-1.173`$ to $`+0.494`$, minimum $`p=0.243`$).

#### The pre/post contrast.

The Fisher exact test on below/above $`\times`$ pre/post counts in the $`\pm 0.5`$-dex window — table $`[[13, 8], [47, 5]]`$ — gives OR $`=0.173`$ (95% CI $`[0.048, 0.619]`$, $`p=0.00715`$). At the pre-Act above:below ratio, $`28.9`$ post-Act models are expected in $`(25.0, 25.5]`$; 5 are observed, so roughly 24 models are “missing” — an $`82.7\%`$ deficit. The formal interaction between the period-specific density jumps is a null at the primary specification and is stated as one: $`\Delta\theta=-0.454`$ (SE $`1.311`$, $`z=-0.346`$, $`p=0.729`$). Across the grid the interaction is always negative at $`h\geq 0.75`$ but significant only at $`h=1.0`$ with 12/16/20 bins ($`p=0.0249`$/$`0.0049`$/$`0.0342`$; $`h=1.0`$ with 10 bins gives $`p=0.0639`$), where curvature leakage is a concern (Section <a href="#sec:methods" data-reference-type="ref" data-reference="sec:methods">3</a>). The pre-period, with only 21–47 in-window models, is too small to pin down the counterfactual discontinuity.

#### Placebos.

The deficit is specific to the statutory line and period. At placebo threshold $`24.5`$ the same pre/post Fisher contrast gives OR $`=1.637`$, $`p=0.248`$; restricted to the post-GPAI period it gives OR $`=2.510`$, $`p=0.0386`$ with the *opposite* sign — more mass in $`[24.5, 25.0)`$ than $`[24.0, 24.5)`$, i.e. mass piling up just below $`25.0`$, consistent with bunching at the line and inconsistent with a generic high-compute slowdown. At placebo threshold $`25.5`$: OR $`=4.00`$, $`p=0.350`$, direction opposite to the $`25.0`$ effect. The placebo split date 2023-08-01, with both halves pre-Act, gives OR $`=0.90`$, $`p=1.0`$ at the true threshold — no pseudo-effect.

#### Timing.

Below:above counts per half-year in the $`\pm 0.5`$-dex window are 2023H1 2:2, 2023H2 3:1, 2024H1 6:4, 2024H2 11:3, then 2025H1 17:1, 2025H2 15:1, 2026H1 6:1 (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). The collapse begins in 2025H1 — after entry into force (August 2024), before the obligations became applicable (August 2025) — consistent with anticipatory compliance: the $`10^{25}`$ presumption was public from July 2024 and attaches to models placed on the market from August 2025. By year, $`[25.0, 25.5)`$ holds 3, 7, 2, 1 models in 2023–2026 while $`[24.5, 25.0)`$ grows 5, 17, 32, 6. Secular compute growth biases *against* the finding — later cohorts should put more, not less, mass above any fixed line, and do at $`24.5`$. Meanwhile 13 post-Act models sit at or above $`10^{25}`$ at some height, with the frontier jumping far above the notch (Grok 3 at $`\log_{10}=26.54`$, GPT-4.5 at $`26.58`$, Llama 4 Behemoth preview at $`25.71`$, Grok 4 at $`26.70`$, GPT-5 at $`25.82`$) — the classic bunching signature in which a notch distorts only marginal decisions .

<table id="tab:main">
<caption>Headline estimates. Density tests: binned-Poisson jump <span class="math inline"><em>θ</em></span> in log density at the statutory cutoff <span class="math inline">log<sub>10</sub>FLOP = 25.0</span>; the primary window is <span class="math inline">±0.5</span> dex with 10 bins, and the sensitivity grid crosses <span class="math inline"><em>h</em> ∈ {0.5, 0.75, 1.0}</span> with <span class="math inline">{8, 10, 12, 16, 20}</span> bins. Fisher tests use below/above counts in the <span class="math inline">±0.5</span>-dex window by period.</caption>
<thead>
<tr>
<th style="text-align: left;">Quantity</th>
<th style="text-align: right;">Estimate</th>
<th style="text-align: right;">SE / CI</th>
<th style="text-align: right;"><span class="math inline"><em>p</em></span></th>
<th style="text-align: left;">notes</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: density jump at the statutory line</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>θ</em></span>, pre-Act (2023-01..2024-07)</td>
<td style="text-align: right;"><span class="math inline">−0.880</span></td>
<td style="text-align: right;"><span class="math inline">0.973</span></td>
<td style="text-align: right;"><span class="math inline">0.366</span></td>
<td style="text-align: left;"><span class="math inline"><em>n</em> = 21</span>, 13/8; 0/15 specs <span class="math inline"><em>p</em> &lt; 0.05</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>θ</em></span>, post-Act (2024-08..)</td>
<td style="text-align: right;"><span class="math inline">−1.334</span></td>
<td style="text-align: right;"><span class="math inline">0.878</span></td>
<td style="text-align: right;"><span class="math inline">0.129</span></td>
<td style="text-align: left;"><span class="math inline"><em>n</em> = 52</span>, 47/5; 10/15 specs <span class="math inline"><em>p</em> &lt; 0.05</span></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>Δ</em><em>θ</em></span> (post <span class="math inline">−</span> pre)</td>
<td style="text-align: right;"><span class="math inline">−0.454</span></td>
<td style="text-align: right;"><span class="math inline">1.311</span></td>
<td style="text-align: right;"><span class="math inline">0.729</span></td>
<td style="text-align: left;"><em>null at primary spec</em></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: Fisher exact below/above <span class="math inline">×</span> pre/post</em></td>
</tr>
<tr>
<td style="text-align: left;">threshold <span class="math inline">25.0</span>, Act date</td>
<td style="text-align: right;">OR <span class="math inline">0.173</span></td>
<td style="text-align: right;"><span class="math inline">[0.048, 0.619]</span></td>
<td style="text-align: right;"><span class="math inline">0.00715</span></td>
<td style="text-align: left;"><span class="math inline">[[13, 8], [47, 5]]</span>; <span class="math inline">−82.7%</span></td>
</tr>
<tr>
<td style="text-align: left;">placebo threshold <span class="math inline">24.5</span></td>
<td style="text-align: right;">OR <span class="math inline">1.637</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.248</span></td>
<td style="text-align: left;">post-GPAI: OR <span class="math inline">2.510</span>, <span class="math inline"><em>p</em> = 0.0386</span></td>
</tr>
<tr>
<td style="text-align: left;">placebo threshold <span class="math inline">25.5</span></td>
<td style="text-align: right;">OR <span class="math inline">4.00</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.350</span></td>
<td style="text-align: left;">opposite direction</td>
</tr>
<tr>
<td style="text-align: left;">placebo date 2023-08-01</td>
<td style="text-align: right;">OR <span class="math inline">0.90</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">1.0</span></td>
<td style="text-align: left;">both halves pre-Act; no effect</td>
</tr>
</tbody>
</table>

Notes: $`\theta<0`$ is a density deficit just above the cutoff. SEs are Wald standard errors from the binned Poisson fit. Significant sensitivity $`p`$-values for the post-Act $`\theta`$ range from $`0.0010`$ to $`0.0143`$ (all $`h=0.75`$ and $`h=1.0`$ specifications); the interaction is significant only at $`h=1.0`$ with $`\geq 12`$ bins ($`p=0.0049`$–$`0.0342`$). Placebo-threshold odds ratios use the same $`\pm 0.5`$-dex construction centered on the placebo line.

# Caveats and Conclusion

Three caveats bound the claims. First, *small n*: only 5 models sit in $`(25.0, 25.5]`$ and 13 at any height above $`10^{25}`$ post-Act; the preferred narrow-window pre/post interaction is $`p=0.729`$, and the interaction is significant only at $`h=1.0`$ with $`\geq 12`$ bins ($`p`$ from $`0.0049`$ to $`0.0342`$), where curvature of the steeply falling density can leak into $`\theta`$. Second, *measurement, not behavior*: post-2024 frontier compute values are largely Epoch estimates, and models without a compute estimate are excluded (719 of $`\approx 3{,}500`$ rows have compute). If the Act suppresses *disclosure* of parameters and token counts near the line, the identical density gap appears with no training-compute response at all; distinguishing a compliance response from a reporting response requires disclosure-side data and is the designated follow-up. Third, a *confounded trend break*: the deficit’s onset (2025H1) coincides with the industry shift toward inference-time scaling and distilled or mixture-of-experts models that followed the late-2024 o1-style reasoning models, which independently reduced demand for $`10^{25}`$-plus pre-training runs among non-frontier labs — though this confound predicts a broad high-compute slowdown, not a deficit specific to the statutory line with mass accumulating immediately below it.

The conclusion is graded accordingly. The post-Act deficit just above the EU AI Act’s $`10^{25}`$-FLOP line is sign-consistent in all 15 density specifications, large (Fisher $`p=0.00715`$, $`82.7\%`$ missing mass), specific to the statutory threshold and period, and correctly timed, with the marginal-crossers-vanish, frontier-jumps-far-above signature that notch responses produce. But with 5–13 post-Act models above the line, a null preferred narrow-window interaction, and an Act-induced disclosure response that would mimic bunching exactly, it is moderate evidence of bunching — behavioral or reporting — rather than proof of a training-compute response. Either reading is informative for the compute-threshold debate : the forward projections of assumed threshold-blind scaling, and the data just above the line already look threshold-aware.

#### Reproducibility.

All estimates derive deterministically from the public Epoch AI panel; no dataset files are committed. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates from the committed `figures/make_fig.py`, which asserts the headline estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Epoch AI (2026). Data on AI models. <https://epoch.ai/data/ai-models> (compute panel extracted June 2026).

European Union (2024). Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *Official Journal of the European Union*, L series, 12 July 2024.

Heim, L., and Koessler, L. (2024). Training compute thresholds: Features and functions in AI regulation. arXiv:2405.10799.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation*. Software. <https://github.com/HaukeHillebrandt/natex>

Hooker, S. (2024). On the limitations of compute thresholds as a governance strategy. arXiv:2407.05694.

Kleven, H. J. (2016). Bunching. *Annual Review of Economics*, 8, 435–464.

Kleven, H. J., and Waseem, M. (2013). Using notches to uncover optimization frictions and structural elasticities: Theory and evidence from Pakistan. *Quarterly Journal of Economics*, 128(2), 669–723.

Kumar, I., and Manning, S. (2025). Trends in frontier AI model count: A forecast to 2028. arXiv:2504.16138.

McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.

Saez, E. (2010). Do taxpayers bunch at kink points? *American Economic Journal: Economic Policy*, 2(3), 180–212.

Sastry, G., Heim, L., Belfield, H., Anderljung, M., Brundage, M., Hazell, J., O’Keefe, C., Hadfield, G. K., et al. (2024). Computing power and the governance of artificial intelligence. arXiv:2402.08797.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
