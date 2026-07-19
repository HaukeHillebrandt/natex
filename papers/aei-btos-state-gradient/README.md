> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/aei-btos-state-gradient/) · [PDF in this repo](./main.pdf)

# Introduction

The release of the DeepSeek-R1 reasoning model on 20 January 2025 is the natural candidate event for a break in US business AI adoption: a companion sector-level analysis of the Census Bureau’s Business Trends and Outlook Survey (BTOS) found a sharp acceleration in AI-exposed sectors at exactly the first post-R1 survey wave, but attribution to R1 failed a placebo battery. This paper asks the cross-sectional version of the same question: did *states* whose workforces use AI more intensively accelerate measured business AI adoption differentially at R1? The design — state-level BTOS adoption interacted with an external AI-exposure measure — was excluded from an earlier automated screen for want of exposure weights; here we revive it with real ones.

Both sides of the design rest on new measurement. The outcome is the BTOS AI use rate, a biweekly probability-sample estimate of the share of US firms using AI, introduced and validated by . The exposure is the Anthropic Economic Index (AEI), which maps Claude conversations to economic tasks ; its geographic report publishes a usage-per-capita index by US state, which we take as the cross-state exposure measure. Interacting differential exposure with a common shock is the logic that shift-share (Bartik) designs formalize , and its known failure modes discipline the analysis: exposure here is measured *after* the outcome window, so every estimate below is a descriptive gradient, never a causal Bartik coefficient; pre-existing differential trends can masquerade as treatment effects ; and slope changes at a single date in smooth aggregate series are spuriously significant often enough that placebo distributions at non-event dates, in the spirit of , are the appropriate yardstick. The per-state slope-change contrast is a kink-style estimand in the sense of the difference-in-kinks design , estimated and falsified with the `natex` toolkit , whose scan-based DiD supplies a localization check. The battery converts one reproduced headline gradient and three nominally significant side results into a clean null plus three identified artifacts.

# Data

The outcome panel combines three BTOS sources. The state sheets of the AI core workbook (DRB approval CBDRB-FY25-0425) give biweekly estimates and standard errors of the share answering “Yes” to Question 7 (“In the last two weeks, did this business use Artificial Intelligence (AI) in producing goods or services?”), waves 202319–202520. A fresh census.gov download extends the panel under the December-2025 rewording (“…in any of its business activities?”), waves 202524–202614; new-wording values are divided by $`1.553`$, the national new/old measurement-RDD ratio estimated at the rewording cutoff by a companion paper , with an additive $`-6.04`$ pp variant as a diagnostic. The archived remote-work Question 6 (share of businesses with paid employees working from home at least one workday), waves 202417–202514, supplies a placebo outcome (1,221 state-wave cells). The assembled panel has 2,727 state-wave cells across 51 states (50 plus DC) and 71 waves, $`t=2023.710`$–$`2026.507`$ in fractional years of collection-period midpoints; $`24.7\%`$ of the state-by-wave grid is disclosure-suppressed, concentrated in small states.

Exposure is the AEI `usage_per_capita_index` per state , averaged over the 2026-04-01..05-01 and 2026-05-01..06-01 data windows (verified: CA $`=(1.56+1.62)/2=1.59`$); it ranges from $`3.52`$ (DC) to $`0.245`$ (WV). Two features bind the interpretation. First, the exposure window (April–June 2026) lies entirely *after* the outcome window, so exposure is endogenous by construction and no estimate below is causal, whatever its $`p`$-value. Second, the index measures Claude usage specifically, proxying all-AI exposure with attenuation and tech-hub confounding both plausible.

# Design and Methods

#### Exposure gradient at a known cutoff.

For each state $`s`$, local OLS slopes of the adoption share in calendar time are fit on each side of $`t_0=2025.055`$ (DeepSeek-R1; first post wave 202503) within $`\pm w`$ years, and the slope change $`\Delta\kappa_s`$ is the post-minus-pre difference — the right-minus-left kink convention of , with the cross-state gradient replacing the group contrast. The headline statistic is the cross-state regression of $`\Delta\kappa_s`$ on AEI exposure, $`b`$ in pp/yr per index unit. The spec of record is $`w=0.5`$, unweighted, restricted to states with at least 6 usable waves per side ($`n=36`$; DC never enters — it is suppressed below the wave minimum). Inference is primary by exposure permutation (AEI values reassigned across states, $`B=10{,}000`$, `default_rng(0)`), with HC1 standard errors reported alongside ; windows $`w\in\{0.5, 0.75, 1.0, 1.4\}`$ are examined, and an SE-weighted variant fits $`1/\mathrm{se}^2`$ WLS slopes from the published State SE sheets and combines them by inverse-variance meta-analysis.

#### Falsification battery.

\(i\) placebo cutoffs at the non-event dates $`2024.30`$ and $`2024.50`$, at o1 ($`2024.6967`$, a real secondary event), at $`2025.30`$, and at $`2025.555`$ (a window that straddles the rewording splice); (ii) the placebo outcome (remote work) at the R1 cutoff — noting that the federal return-to-office executive order was signed on 2025-01-20, the *same day* as R1, so remote-work breaks at that date are expected for non-AI reasons; (iii) leverage checks dropping DC and CA, and a Spearman rank gradient; (iv) splice diagnostics rerunning the long-window gradients under the multiplicative ($`/1.553`$) splice, the additive ($`-6.04`$ pp) splice, and old-wording-only data ($`t<2025.8`$).

#### Panel estimators.

A tercile two-way fixed-effects (TWFE) level DiD interacts the top AEI tercile (17 states, cut $`1.002`$) with post-R1, with wave and state fixed effects, CR1 standard errors by state , a placebo-in-space permutation of the treated set ($`B=2{,}000`$), and a state-specific linear-trends variant. The `natex` survey pipeline’s scan-based DiD (SuDDDS) searches break dates and windows on the same panel (seed 0, normal model, AR(1)-by-unit permutation null) as a localization and role-assignment check.

#### Honest-inference notes, stated as run.

HC1/CR1 intervals on smooth biweekly aggregates are potentially oversized, so permutation $`p`$ is the primary inference throughout. The `natex` refusals are reported, not patched over: the SuDDDS anticipation test refused (Holm $`p`$ all NaN), and all three effect estimators (dd, synthetic control, GESS) refused with “only 0 usable placebos; $`\geq 5`$ required” on this unbalanced panel — the manual TWFE and permutation estimates below are the effect numbers of record.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) State-level slope change in BTOS AI use at DeepSeek-R1 (<span class="math inline"><em>w</em> = ±0.5</span> yr, <span class="math inline"><em>n</em> = 36</span> states with <span class="math inline"> ≥ 6</span> waves per side) against AEI usage-per-capita exposure, with the unweighted OLS fit: <span class="math inline"><em>b</em> = +1.635</span> pp/yr per index unit, exposure-permutation <span class="math inline"><em>p</em> = 0.257</span>. (b) The same gradient statistic at non-event placebo cutoffs, at o1, at R1, and for the remote-work placebo outcome at R1, with 95% HC1 intervals; filled markers are nominally significant by permutation at the 5% level. The only filled marker in the battery is a <em>placebo</em> — the SE-weighted remote-work gradient at R1 (<span class="math inline">+4.39</span>, <span class="math inline"><em>p</em> = 0.045</span>).</figcaption>
</figure>

#### The scout number reproduces exactly, and it is null.

At the spec of record ($`w=0.5`$, unweighted, 36 states), the R1 slope-change gradient is $`b=+1.635`$ pp/yr per AEI index unit (HC1 se $`1.458`$; classical se $`1.428`$, matching the scout’s “se 1.43, $`t`$ 1.15”; $`t_{\mathrm{HC1}}=1.12`$; correlation $`0.193`$), with exposure-permutation $`p=0.257`$ (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>a). The SE-weighted version is $`+1.877`$ (HC1 se $`1.922`$, $`p_{\mathrm{perm}}=0.360`$); the Spearman rank gradient is $`\rho=0.098`$ ($`p=0.568`$); dropping DC and CA gives $`+1.54`$ ($`p=0.326`$) unweighted and $`+0.53`$ ($`p=0.804`$) weighted. DC — the highest-exposure unit at AEI $`3.52`$ — never enters the gradient sample in the first place, so no single unit drives the (non-)result.

#### Placebos bracket the headline.

The identical statistic at non-event cutoffs returns $`+1.56`$ ($`p=0.277`$) at $`t=2024.30`$ and $`+2.24`$ ($`p=0.162`$) at $`t=2024.50`$ — magnitudes that *bracket* the R1 estimate — and $`-0.23`$ ($`p=0.889`$) at o1 (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel B; Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). The remote-work placebo outcome at R1 is $`+1.97`$ ($`p=0.288`$), the same magnitude as the headline; its SE-weighted version is $`+4.39`$ (HC1 se $`1.895`$) with permutation $`p=0.045`$ — a nominally significant *placebo*. A design that produces R1-sized gradients at dates where nothing happened, and a significant gradient for an outcome whose own policy shock (the federal return-to-office executive order, signed 2025-01-20) merely shares R1’s date, manufactures gradients; the null at R1 carries no information beyond that.

#### Artifact 1: the long-window sign flip is the splice.

Once the window reaches across the December-2025 rewording, the gradient turns significantly negative: $`-4.31`$ ($`p=0.011`$) at $`w=1.0`$ and $`-2.53`$ ($`p=0.011`$) at $`w=1.4`$ (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel C). This is a measurement artifact, identified three ways. AEI exposure correlates $`+0.676`$ with the pre-R1 adoption level, so the nationally calibrated multiplicative ratio $`1.553`$ misprices high-level (equivalently high-AEI) states; the additive $`-6.04`$ pp splice — wrong in the opposite direction — attenuates the same estimates to $`-2.92`$ and $`-1.07`$; and old-wording-only data collapse both to null ($`-0.87`$, $`p=0.279`$ at $`w=1.0`$; $`+0.44`$, $`p=0.513`$ at $`w=1.4`$). A placebo cutoff at $`2025.555`$, placed so the window straddles the splice with no event nearby, yields $`-9.16`$ ($`p=0.00025`$) multiplicative and still $`-5.81`$ ($`p=0.015`$) additive: *neither* national correction is valid cross-state, so state-level gradients through the rewording — including any June-2026 extension of this design — are unidentified.

#### Artifact 2: the TWFE level effect is a pre-trend.

The tercile TWFE DiD gives $`\hat\tau=+0.527`$ pp (CR1 se $`0.210`$) with placebo-in-space permutation $`p=0.011`$ ($`+0.547`$, $`p=0.038`$ on the full panel) — nominally the strongest result in the study (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel D). But the high-minus-low AEI gap was already growing at $`+0.600`$ pp/yr before R1 (HC1 se $`0.180`$, 36 pre-R1 waves), which mechanically predicts roughly $`+0.3`$ pp of the estimate over the post window, and adding state-specific linear trends flips the $`w=0.5`$ estimate to $`-0.38`$ (se $`0.31`$). This is the textbook pre-trend failure : the level DiD is descriptive only, and the trend-robust estimand — the slope-change gradient — is exactly the null of panel A.

#### Artifact 3: the scan localization is treatment rediscovery.

The SuDDDS scan (seed 0, both panels) maximizes at LLR $`154.5`$ for the empty subset — *all* states — at $`t_0=2025.0877`$, the first post-R1 wave, with scan $`p=0.010`$ under the AR(1)-by-unit null. Because the declared treatment column is the constructed `high_aei`$`\times`$`post` indicator, a scan that lights up for all states at the post boundary is a mechanical rediscovery of that column, not outcome evidence — the same role-assignment failure documented in the companion sector study . The composition check passes ($`p=0.707`$); the anticipation test and all three effect estimators refused, as stated in Section <a href="#sec:methods" data-reference-type="ref" data-reference="sec:methods">3</a>.

<table id="tab:main">
<caption>Exposure gradients, placebos, and panel estimates. Gradient rows: slope-change gradient <span class="math inline"><em>b</em></span> (pp/yr per AEI index unit) with HC1 se and exposure-permutation <span class="math inline"><em>p</em></span> (<span class="math inline"><em>B</em> = 10, 000</span>; splice diagnostics <span class="math inline"><em>B</em> = 4, 000</span>). TWFE rows: level effect (pp) with CR1 se by state and placebo-in-space permutation <span class="math inline"><em>p</em></span> (<span class="math inline"><em>B</em> = 2, 000</span>).</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;">Estimate</th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline"><em>p</em><sub>perm</sub></span></th>
<th style="text-align: right;"><span class="math inline"><em>n</em></span></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: AEI gradient at DeepSeek-R1 (<span class="math inline"><em>w</em> = 0.5</span> yr)</em></td>
</tr>
<tr>
<td style="text-align: left;">unweighted (spec of record)</td>
<td style="text-align: right;"><span class="math inline">+1.635</span></td>
<td style="text-align: right;"><span class="math inline">1.458</span></td>
<td style="text-align: right;"><span class="math inline">0.257</span></td>
<td style="text-align: right;">36</td>
</tr>
<tr>
<td style="text-align: left;">SE-weighted (WLS <span class="math inline">+</span> IV meta)</td>
<td style="text-align: right;"><span class="math inline">+1.877</span></td>
<td style="text-align: right;"><span class="math inline">1.922</span></td>
<td style="text-align: right;"><span class="math inline">0.360</span></td>
<td style="text-align: right;">36</td>
</tr>
<tr>
<td style="text-align: left;">unweighted, drop DC<span class="math inline">+</span>CA</td>
<td style="text-align: right;"><span class="math inline">+1.544</span></td>
<td style="text-align: right;"><span class="math inline">1.651</span></td>
<td style="text-align: right;"><span class="math inline">0.326</span></td>
<td style="text-align: right;">35</td>
</tr>
<tr>
<td style="text-align: left;">SE-weighted, drop DC<span class="math inline">+</span>CA</td>
<td style="text-align: right;"><span class="math inline">+0.531</span></td>
<td style="text-align: right;"><span class="math inline">2.316</span></td>
<td style="text-align: right;"><span class="math inline">0.804</span></td>
<td style="text-align: right;">35</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: placebo cutoffs and placebo outcome (<span class="math inline"><em>w</em> = 0.5</span>, unweighted)</em></td>
</tr>
<tr>
<td style="text-align: left;">placebo cutoff <span class="math inline"><em>t</em><sub>0</sub> = 2024.30</span></td>
<td style="text-align: right;"><span class="math inline">+1.560</span></td>
<td style="text-align: right;"><span class="math inline">1.224</span></td>
<td style="text-align: right;"><span class="math inline">0.277</span></td>
<td style="text-align: right;">33</td>
</tr>
<tr>
<td style="text-align: left;">placebo cutoff <span class="math inline"><em>t</em><sub>0</sub> = 2024.50</span></td>
<td style="text-align: right;"><span class="math inline">+2.238</span></td>
<td style="text-align: right;"><span class="math inline">1.604</span></td>
<td style="text-align: right;"><span class="math inline">0.162</span></td>
<td style="text-align: right;">34</td>
</tr>
<tr>
<td style="text-align: left;">o1 cutoff <span class="math inline"><em>t</em><sub>0</sub> = 2024.6967</span></td>
<td style="text-align: right;"><span class="math inline">−0.233</span></td>
<td style="text-align: right;"><span class="math inline">1.610</span></td>
<td style="text-align: right;"><span class="math inline">0.889</span></td>
<td style="text-align: right;">33</td>
</tr>
<tr>
<td style="text-align: left;">remote work (Q6) at R1</td>
<td style="text-align: right;"><span class="math inline">+1.970</span></td>
<td style="text-align: right;"><span class="math inline">1.458</span></td>
<td style="text-align: right;"><span class="math inline">0.288</span></td>
<td style="text-align: right;">51</td>
</tr>
<tr>
<td style="text-align: left;">remote work (Q6) at R1, SE-weighted</td>
<td style="text-align: right;"><span class="math inline">+4.392</span></td>
<td style="text-align: right;"><span class="math inline">1.895</span></td>
<td style="text-align: right;"><span class="math inline">0.045<sup>*</sup></span></td>
<td style="text-align: right;">51</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: long windows across the 2025-12 rewording (unweighted)</em></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>w</em> = 1.0</span>, spliced (<span class="math inline">/1.553</span>)</td>
<td style="text-align: right;"><span class="math inline">−4.305</span></td>
<td style="text-align: right;"><span class="math inline">1.676</span></td>
<td style="text-align: right;"><span class="math inline">0.011<sup>*</sup></span></td>
<td style="text-align: right;">39</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>w</em> = 1.0</span>, additive splice (<span class="math inline">−6.04</span> pp)</td>
<td style="text-align: right;"><span class="math inline">−2.922</span></td>
<td style="text-align: right;"><span class="math inline">1.885</span></td>
<td style="text-align: right;"><span class="math inline">0.020<sup>*</sup></span></td>
<td style="text-align: right;">39</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>w</em> = 1.0</span>, old wording only</td>
<td style="text-align: right;"><span class="math inline">−0.873</span></td>
<td style="text-align: right;"><span class="math inline">1.000</span></td>
<td style="text-align: right;"><span class="math inline">0.279</span></td>
<td style="text-align: right;">39</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>w</em> = 1.4</span>, spliced</td>
<td style="text-align: right;"><span class="math inline">−2.535</span></td>
<td style="text-align: right;"><span class="math inline">0.945</span></td>
<td style="text-align: right;"><span class="math inline">0.011<sup>*</sup></span></td>
<td style="text-align: right;">40</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><em>w</em> = 1.4</span>, old wording only</td>
<td style="text-align: right;"><span class="math inline">+0.439</span></td>
<td style="text-align: right;"><span class="math inline">0.645</span></td>
<td style="text-align: right;"><span class="math inline">0.513</span></td>
<td style="text-align: right;">40</td>
</tr>
<tr>
<td style="text-align: left;">splice-window placebo <span class="math inline"><em>t</em><sub>0</sub> = 2025.555</span>, spliced</td>
<td style="text-align: right;"><span class="math inline">−9.161</span></td>
<td style="text-align: right;"><span class="math inline">2.150</span></td>
<td style="text-align: right;"><span class="math inline">0.00025<sup>*</sup></span></td>
<td style="text-align: right;">39</td>
</tr>
<tr>
<td style="text-align: left;">splice-window placebo, additive</td>
<td style="text-align: right;"><span class="math inline">−5.806</span></td>
<td style="text-align: right;"><span class="math inline">2.639</span></td>
<td style="text-align: right;"><span class="math inline">0.015<sup>*</sup></span></td>
<td style="text-align: right;">39</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel D: tercile TWFE level DiD (pp)</em></td>
</tr>
<tr>
<td style="text-align: left;">TWFE, <span class="math inline"><em>w</em> = 0.5</span></td>
<td style="text-align: right;"><span class="math inline">+0.527</span></td>
<td style="text-align: right;"><span class="math inline">0.210</span></td>
<td style="text-align: right;"><span class="math inline">0.011<sup>*</sup></span></td>
<td style="text-align: right;">964</td>
</tr>
<tr>
<td style="text-align: left;">TWFE, full panel</td>
<td style="text-align: right;"><span class="math inline">+0.547</span></td>
<td style="text-align: right;"><span class="math inline">0.236</span></td>
<td style="text-align: right;"><span class="math inline">0.038<sup>*</sup></span></td>
<td style="text-align: right;">2727</td>
</tr>
<tr>
<td style="text-align: left;">TWFE <span class="math inline">+</span> state trends, <span class="math inline"><em>w</em> = 0.5</span></td>
<td style="text-align: right;"><span class="math inline">−0.382</span></td>
<td style="text-align: right;"><span class="math inline">0.315</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">964</td>
</tr>
<tr>
<td style="text-align: left;">pre-R1 high-minus-low gap slope (pp/yr)</td>
<td style="text-align: right;"><span class="math inline">+0.600</span></td>
<td style="text-align: right;"><span class="math inline">0.180</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">36 waves</td>
</tr>
</tbody>
</table>

Notes: $`^{\ast}`$ nominally significant at the 5% level by permutation. The $`w=1.4`$ additive-splice gradient is $`-1.069`$ ($`p=0.095`$). Panel D’s TWFE $`p`$-values are placebo-in-space; the state-trends row and the pre-trend slope row report CR1 and HC1 standard errors respectively, with no permutation test run. Every nominally significant row is attributed to an identified artifact in the text: splice mispricing (panel C), a pre-existing differential trend (panel D), or a same-day policy shock to the placebo outcome (panel B).

# Caveats and Conclusion

Four caveats bound the claims. First, *exposure is post-outcome*: the AEI index is measured April–June 2026, after every outcome wave used at the R1 cutoff, so even a robust gradient would have been descriptive, not Bartik-causal — the null is about this gradient design, not proof of zero differential adoption. Second, *exposure is Claude-specific*: as a proxy for all-AI exposure it is attenuated and plausibly confounded with tech-hub geography. Third, *coverage*: $`24.7\%`$ of the state-wave grid is disclosure-suppressed, concentrated in small states; estimation samples run 33–40 states, and DC — the most exposed unit — is never estimable at the primary window. Fourth, *the calendar*: R1’s release week was inauguration week, sharing its date with the federal return-to-office executive order (2025-01-20) and adjoining the Stargate announcement (2025-01-21), the AI-diffusion export rule (2025-01-13), and a BTOS sample-year rollover, so even a surviving gradient could not have been attributed to R1 alone.

The conclusion is a disciplined null. The headline gradient reproduces exactly and is statistically indistinguishable from the same statistic at dates where nothing happened; the one nominally significant gradient in the $`w=0.5`$ battery belongs to a placebo outcome with its own same-day shock; and all three seemingly significant side results — the long-window negative gradients, the TWFE level effect, and the scan localization — dissolve under, respectively, old-wording-only data, state trends, and role-assignment scrutiny. These data provide no credible evidence that high-AEI-exposure states accelerated business AI adoption differentially at DeepSeek-R1, and the state-level splice problem implies that extending the design through the rewording cannot rescue it.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 at seed 0 from the frozen BTOS workbooks (published at <https://www.census.gov/hfp/btos>) and the public AEI state release ; no dataset files are committed. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline gradient and the significant-placebo numbers against the numbers of record before drawing.

<div class="thebibliography">

9

Appel, R., McCrory, P., Tamkin, A., McCain, M., Neylon, T., and Stern, M. (2025). Anthropic Economic Index report: Uneven geographic and enterprise AI adoption. arXiv:2511.15080.

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bonney, K., Breaux, C., Buffington, C., Dinlersoz, E., Foster, L., Goldschlag, N., Haltiwanger, J., Kroff, Z., and Savage, K. (2024). Tracking firm use of AI in real time: A snapshot from the Business Trends and Outlook Survey. NBER Working Paper No. 32319.

Cameron, A. C., and Miller, D. L. (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Goldsmith-Pinkham, P., Sorkin, I., and Swift, H. (2020). Bartik instruments: What, when, why, and how. *American Economic Review*, 110(8), 2586–2624.

Guo, D., Yang, D., Zhang, H., et al. (2025). DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. *Nature*, 645(8081), 633–638.

Handa, K., Tamkin, A., McCain, M., et al. (2025). Which economic tasks are performed with AI? Evidence from millions of Claude conversations. arXiv:2503.04761.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

Hillebrandt, H. (2026). A sharp bend, no verdict: Difference-in-kinks tests of US business AI adoption at DeepSeek-R1. Companion paper, natex paper collection. <https://haukehillebrandt.github.io/natex/btos-sector-did-r1/>

Hillebrandt, H. (2026). The question is the treatment: A known-cutoff regression discontinuity at the BTOS AI-question rewording. Companion paper, natex paper collection. <https://haukehillebrandt.github.io/natex/btos-rewording-rdd/>

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

Roth, J. (2022). Pretest with caution: Event-study estimates after testing for parallel trends. *American Economic Review: Insights*, 4(3), 305–322.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
