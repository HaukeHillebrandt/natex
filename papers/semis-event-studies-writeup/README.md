> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/semis-event-studies-writeup/) · [PDF in this repo](./main.pdf)

# Introduction

On 7 October 2022 the US Bureau of Industry and Security (BIS) imposed broad export controls on advanced AI accelerators bound for China, read at the time as a landmark attempt to restrict China’s frontier compute ; on 17 October 2023 a second round closed the A800/H800 loophole. Export controls impose costs on the controlling country’s own firms : entity-list evidence puts US suppliers’ collateral losses near \$130 billion in market value, plus lost customers, lending, and employment . On 27 January 2025 a demand-side shock arrived from the opposite direction: the release of DeepSeek’s R1 model triggered the largest one-day market-capitalization loss in history for Nvidia, and event studies document significantly negative abnormal returns across US semiconductor stocks .

The standard tool here is the event study : short windows, abnormal returns against a market model, an efficient-markets identification argument. This paper deliberately uses the wrong tool — quasi-experimental designs with *calendar time* as the running or assignment variable — and asks what an automated toolkit’s validation layer does about it. The dangers are known in principle: regression discontinuity in time lacks the cross-sectional no-manipulation logic of true RDDs and is biased by ignored time-series structure , and a regression kink estimated on a calendar-time running variable is, mechanically, a before/after slope contrast. The open question is empirical: does a modern validation battery — placebo covariates and cutoffs , density tests, bandwidth sensitivity, scan-based localization — actually catch these failure modes on real financial series? We run `natex` over three declared cutoffs and report every verdict, including the refusals. The answer is asymmetric: every “credible” verdict fails inspection, while the refusals, failures, and nulls are the informative outputs.

# Data

Weekly adjusted-close prices for NVDA, AMD, ASML, TSM, and the SOXX semiconductor ETF, 784 weeks from July 2011 to July 2026 (Yahoo Finance weeklies; not committed). The analysis variable is the log adjusted close, the running variable calendar time in years; the pooled five-ticker panel gives the kink family $`n_{\mathrm{used}} = 1960`$. Descriptive contrasts use the NVDA-minus-SOXX relative log price, netting out the sector factor. The three declared cutoffs are $`t = 2022.7644`$ (BIS round one, 2022-10-07), $`t = 2023.7918`$ (BIS round two, 2023-10-17), and $`t = 2025.0712`$ (the DeepSeek crash week beginning 2025-01-27). One confound matters throughout: ChatGPT’s release (2022-11-30) lands eight weeks *after* the first BIS cutoff, inside every bandwidth the 2022 analyses select.

# Design and Methods

Three `natex` survey runs (v0.2.0, seed 0) supply the estimates: a pooled five-ticker run declaring the DeepSeek cutoff, a pooled run declaring the BIS-2022 cutoff, and an NVDA-only run declaring the BIS-2023 cutoff. Each survey attempts seven design families on the same panel: a declared regression kink in time (local-linear RKD in the difference-in-kinks tradition of , cross-validated bandwidth, Holm-corrected placebo battery in the spirit of ), a discovered RDD (candidate role assignment over {treatment, forcing, outcome} columns, scan localization , McCrary-style density test, covariate placebos), a sector-by-time DiD scan (SuDDDS, permutation-calibrated), synthetic control, discovered-experiments estimation (dee), IV, and bunching.

#### Honest-inference notes, stated as run.

These caveats are part of the record, not afterthoughts. (i) `natex`’s own kink report carries the warning that “a kink in a calendar-time running variable is a before/after slope contrast” — the causal reading is on the analyst, and this paper’s point is what happens when it is taken anyway. (ii) The DeepSeek-week level break below is descriptive: no placebo distribution was run, so it carries no standard error. (iii) The DeepSeek-run DiD *failed* rather than returning a number: the intake ignored the declared numeric time column and picked the string `date` column (error of record: “`time column must be numeric: date`”). (iv) The NVDA-only synthetic control *failed* by construction — “`treated unit has no finite outcome before t0`” — a single-ticker panel has no donors. (v) The dee family returned nulls with zero usable discovered experiments (fewer than its minimum of three); IV and bunching declined as `needs_input` throughout; and where a placebo battery was vacuous we say so and treat the estimate as unvalidated.

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Weekly NVDA-minus-SOXX relative log (adjusted-close) price with the three declared cutoffs; ChatGPT lands eight weeks after the BIS-2022 cutoff, and the 2023–2025 AI rally dominates the series. (b) The DeepSeek week: an <span class="math inline">11.6</span> log-point one-week relative level break with locally unchanged <span class="math inline">±26</span>-week slope — a level break, not a kink. (c) Declared-kink estimates (95% CIs) at half, headline (diamond, cross-validated), and double bandwidth: BIS-2022 collapses monotonically, BIS-2023 fades to <span class="math inline"><em>t</em> ≈ 1.06</span>, and the DeepSeek “kink” is positive at a crash — the rally on both sides.</figcaption>
</figure>

# Results

#### The real event: a relative level break at DeepSeek.

Recomputed exactly from the weekly closes, the week beginning 2025-01-27 took NVDA down $`-0.17211`$ in log terms ($`-17.2`$ log-points; $`-15.8\%`$ simple return) against $`-0.05577`$ for SOXX ($`-5.6`$ log-points; $`-5.4\%`$): an $`-11.63`$ log-point one-week fall in NVDA *relative to its own sector*. The local relative trend barely moved: $`\pm 26`$-week OLS slopes of the relative log price are $`+0.52`$ log/yr before and $`+0.39`$ log/yr after (Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>b). The DeepSeek shock is a *level* break with an approximately unchanged local slope — the signature of a one-off repricing, not a growth-regime change. This estimate is descriptive (no placebo distribution; no standard error).

#### Declared kinks: “credible” three for three, wrong three for three.

Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A. At the DeepSeek cutoff the pooled kink is $`\hat\tau = +0.4756`$ log/yr (se $`0.0839`$, 95% CI $`[0.3111, 0.6400]`$, Holm $`p = 1.4\times 10^{-8}`$, cross-validated bandwidth $`6.05`$ yr, $`n_{\mathrm{used}} = 1960`$). The sign alone refutes the causal reading: DeepSeek was a crash, yet the “effect” is a large *positive* slope change — a six-year bandwidth spans the pre-2023 flat regime on the left and the AI rally on the right, so the kink dates the rally, not the event. At the BIS-2022 cutoff the pooled kink ($`+0.1305`$, Holm $`p = 0.022`$) is monotonically bandwidth-sensitive — $`+0.832`$ (se $`0.164`$) at $`1.88`$ yr, $`+0.131`$ (se $`0.057`$) at $`3.75`$ yr, $`+0.050`$ (se $`0.032`$) at $`7.50`$ yr — and ChatGPT’s release sits eight weeks post-cutoff inside every one of those windows; nothing localizes the bend to the export controls. The NVDA-only BIS-2023 kink ($`+0.1534`$, Holm $`p = 0.0011`$) fades to $`+0.0407`$ (se $`0.0385`$, $`t \approx 1.06`$) at double bandwidth. The $`\pm 26`$-week local slopes tell the same story in descriptive form: the relative NVDA-minus-SOXX slope change is $`+1.46`$ log/yr at BIS-2022 (from $`-0.46`$ to $`+1.00`$), $`+0.41`$ at BIS-2023 (from $`+0.61`$ to $`+1.02`$), and $`-0.13`$ ($`\approx`$ flat, $`+0.52 \to +0.39`$) at DeepSeek: the two “export-control kinks” are the AI rally starting, and the one week with a genuine event shows no kink at all.

#### A “credible” RDD that is a role-assignment artifact.

In the BIS-2022 run the discovered-RDD family also returned “credible” (scan $`p = 0.010`$, density $`p = 0.180`$). Inspection of the selected candidate shows why this is empty: the automated role search assigned treatment $`=`$ `post_2022_controls`, forcing $`=`$ `log_adjclose`, and outcome $`=`$ $`t`$ — it mechanically rediscovered that prices above a level tend to lie after 2022. The fuzzy estimate $`\hat\tau_{\mathrm{2SLS}} = 1.581`$ (se $`0.886`$, 95% CI $`[-0.156, 3.317]`$) crosses zero and has no economic content. In the BIS-2023 NVDA-only run the RDD scan rejected at $`p = 0.040`$ but the covariate-placebo battery was *vacuous* (no testable covariate in the single-ticker numeric panel; density $`p = 0.312`$): unvalidated, and not reported as a finding.

#### The informative null: no DiD break at the export controls.

The SuDDDS scan over tickers and break dates in the BIS-2022 run finds nothing at the controls: scan $`p = 0.70`$ (LLR $`2.548`$), with the best-fitting break at $`t_0 = 2025.95`$ (window $`3.76`$ yr) — not $`2022.76`$ (point estimates at that spurious $`t_0`$ in the table notes; scan-level $`p`$-values are not computed for a non-rejected scan). The NVDA-only run’s DiD is likewise null ($`p = 0.76`$, best $`t_0 = 2017.16`$). No ticker-relative break is detectable at either export-control date. This is the substantive result of the paper, and it is a null: whatever costs the controls imposed on these five firms were priced jointly with — and are not separable in these data from — the contemporaneous AI-demand shock lifting the same stocks.

#### The validation layer’s scorecard.

Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel B. Credit: at the DeepSeek cutoff the discovered-RDD family correctly returned *null* (scan $`p = 1.00`$, density $`p = 0.752`$, all eight covariate placebos at Holm $`p = 1.0`$; the fuzzy $`\hat\tau_{\mathrm{2SLS}} = 4.07`$, se $`2.64`$, was discarded), and the pipeline refused rather than fabricated where inputs were missing (`needs_input`; dee nulls; two loud failures quoted in Section <a href="#sec:methods" data-reference-type="ref" data-reference="sec:methods">3</a>). Debit: every “credible” verdict on these series — three kinks and one RDD — fails inspection, and cross-validated bandwidth selection actively favors the artifact, choosing windows wide enough to straddle regimes.

<table id="tab:main">
<caption>All verdicts across the three survey runs (<code>natex</code> v0.2.0, seed 0).</caption>
<thead>
<tr>
<th style="text-align: left;">Run (cutoff)</th>
<th style="text-align: left;">family</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: left;">inference</th>
<th style="text-align: left;">disposition</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6" style="text-align: left;"><em>Panel A: declared kinks in calendar time, log price (log/yr) — all “credible”, none survives</em></td>
</tr>
<tr>
<td style="text-align: left;">BIS-2022 (pooled)</td>
<td style="text-align: left;">kink</td>
<td style="text-align: right;"><span class="math inline">+0.1305</span></td>
<td style="text-align: right;"><span class="math inline">0.0570</span></td>
<td style="text-align: left;">Holm <span class="math inline"><em>p</em> = 0.022</span>; bw 3.75</td>
<td style="text-align: left;"><span class="math inline">+0.83 → +0.13 → +0.05</span> across bw</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2023 (NVDA)</td>
<td style="text-align: left;">kink</td>
<td style="text-align: right;"><span class="math inline">+0.1534</span></td>
<td style="text-align: right;"><span class="math inline">0.0469</span></td>
<td style="text-align: left;">Holm <span class="math inline"><em>p</em> = 0.0011</span>; bw 4.77</td>
<td style="text-align: left;">fades: <span class="math inline"><em>t</em> ≈ 1.06</span> at <span class="math inline">2×</span>bw</td>
</tr>
<tr>
<td style="text-align: left;">DeepSeek (pooled)</td>
<td style="text-align: left;">kink</td>
<td style="text-align: right;"><span class="math inline">+0.4756</span></td>
<td style="text-align: right;"><span class="math inline">0.0839</span></td>
<td style="text-align: left;">Holm <span class="math inline"><em>p</em> = 1.4 × 10<sup>−8</sup></span>; bw 6.05</td>
<td style="text-align: left;">positive “kink” at a crash</td>
</tr>
<tr>
<td colspan="6" style="text-align: left;"><em>Panel B: every other family disposition</em></td>
</tr>
<tr>
<td style="text-align: left;">BIS-2022</td>
<td style="text-align: left;">rdd</td>
<td style="text-align: right;"><span class="math inline">1.581</span></td>
<td style="text-align: right;"><span class="math inline">0.886</span></td>
<td style="text-align: left;">scan <span class="math inline"><em>p</em> = 0.010</span>; CI spans 0</td>
<td style="text-align: left;">role-assignment artifact</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2022</td>
<td style="text-align: left;">did</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">scan <span class="math inline"><em>p</em> = 0.70</span>; best <span class="math inline"><em>t</em><sub>0</sub> = 2025.95</span></td>
<td style="text-align: left;"><strong>informative null</strong></td>
</tr>
<tr>
<td style="text-align: left;">BIS-2022</td>
<td style="text-align: left;">dee</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"><span class="math inline">0 &lt; 3</span> usable experiments</td>
<td style="text-align: left;">null</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2023</td>
<td style="text-align: left;">rdd</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">scan <span class="math inline"><em>p</em> = 0.040</span>; placebos vacuous</td>
<td style="text-align: left;">unvalidated</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2023</td>
<td style="text-align: left;">did</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">scan <span class="math inline"><em>p</em> = 0.76</span>; best <span class="math inline"><em>t</em><sub>0</sub> = 2017.16</span></td>
<td style="text-align: left;">null</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2023</td>
<td style="text-align: left;">sc</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">no donors (single ticker)</td>
<td style="text-align: left;">failed, loudly</td>
</tr>
<tr>
<td style="text-align: left;">BIS-2023</td>
<td style="text-align: left;">dee</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"><span class="math inline">0 &lt; 3</span> usable experiments</td>
<td style="text-align: left;">null</td>
</tr>
<tr>
<td style="text-align: left;">DeepSeek</td>
<td style="text-align: left;">rdd</td>
<td style="text-align: right;"><span class="math inline">4.07</span></td>
<td style="text-align: right;"><span class="math inline">2.64</span></td>
<td style="text-align: left;">scan <span class="math inline"><em>p</em> = 1.00</span>; CI spans 0</td>
<td style="text-align: left;">null (placebos worked)</td>
</tr>
<tr>
<td style="text-align: left;">DeepSeek</td>
<td style="text-align: left;">did</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">intake chose string <code>date</code> col.</td>
<td style="text-align: left;">failed, loudly</td>
</tr>
<tr>
<td style="text-align: left;">DeepSeek</td>
<td style="text-align: left;">iv/sc/bunching</td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"><code>needs_input</code></td>
<td style="text-align: left;">refused</td>
</tr>
</tbody>
</table>

Notes: Kink rows: local-linear RKD in calendar time, log adjusted close, cross-validated bandwidth in years, $`n_{\mathrm{used}}=1960`$; BIS-2022 disposition: estimate at half/headline/double bandwidth. 95% CIs: BIS-2022 kink $`[0.0188,
0.2423]`$; BIS-2023 kink $`[0.0616, 0.2453]`$; DeepSeek kink $`[0.3111,
0.6400]`$. RDD rows: fuzzy 2SLS at the discovered candidate; CIs BIS-2022 $`[-0.156, 3.317]`$ and DeepSeek $`[-1.11, 9.25]`$; density $`p`$: BIS-2022 $`0.180`$, BIS-2023 $`0.312`$, DeepSeek $`0.752`$ with all 8 covariate placebos at Holm $`p=1.0`$. DiD rows: SuDDDS permutation scan; point estimates at the (non-rejected) BIS-2022 best $`t_0`$: $`\hat\tau_{\mathrm{DD}}=-2.061`$ (se $`0.018`$), $`\hat\tau_{\mathrm{synth}}=-0.427`$ (se $`0.025`$).

# Caveats and Conclusion

Five caveats bound the claims. First, a unit correction to the working notes of record: the DeepSeek-week figures $`-17.2\%`$ / $`-5.6\%`$ are one-week *log* returns (log-points); the simple returns are $`-15.8\%`$ and $`-5.4\%`$. Second, the level break and the $`\pm 26`$-week slopes are descriptive — a check on magnitudes against the formal event-study machinery , not a substitute for it. Third, all causal-design caveats for time-as-running-variable apply with full force , and `natex`’s own recorded warning — a kink in a calendar-time running variable is a before/after slope contrast — should be read as governing every panel-A row. Fourth, the event window is confounded by construction: ChatGPT lands eight weeks after the BIS-2022 cutoff, and the 2023–2025 AI rally overlaps everything after it; the DiD null is therefore a statement about non-separability in these data, not evidence that the controls were costless. Fifth, five tickers are a narrow panel; the SuDDDS permutation floor and the vacuous placebo batteries both reflect that thinness.

The conclusion is a methods verdict with one substantive corollary. Methods: on trending financial series, an automated validation layer is asymmetric — its refusals, failures, and nulls were all correct and informative, while every one of its “credible” verdicts was an artifact of trend regimes, role search, or vacuous batteries; cross-validated bandwidths made the kink artifacts *more* confident, not less; “credible” on a calendar-time design is a prompt for the checks in Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a>, not a result. Substance: the one clean event in these data is DeepSeek — an $`11.6`$ log-point one-week repricing of NVDA against its own sector with no local trend change — while the export-control rounds left no detectable ticker-relative break at all, consistent with their costs having been priced against the contemporaneous AI-demand shock rather than as standalone news.

#### Reproducibility.

All estimates come from three `natex` v0.2.0 survey runs at seed 0 (2026-07-18; the kink estimator is deterministic) on frozen weekly price extracts not committed to the repository. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the level break, the local slopes, and the three headline kink estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Allen, G. C. (2022). Choking off China’s access to the future of AI. Center for Strategic and International Studies (CSIS), October 2022. <https://www.csis.org/analysis/choking-chinas-access-future-ai>

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bown, C. P. (2020). Export controls: America’s other national security threat. Peterson Institute for International Economics Working Paper 20-8. <https://www.piie.com/sites/default/files/documents/wp20-8.pdf>

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Crosignani, M., Han, L., Macchiavelli, M., and Silva, A. F. (2025). Securing technological leadership? The cost of export controls on firms. Federal Reserve Bank of New York Staff Report No. 1096 (first circulated 2023 as “Geopolitical Risk and Decoupling: Evidence from U.S. Export Controls”).

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Han, Z. (2025). Silicon disruption: An event study of DeepSeek R1’s breakthrough impact on semiconductor markets. *SHS Web of Conferences*, 218, 01030 (ICDDE 2025). <https://doi.org/10.1051/shsconf/202521801030>

Hausman, C., and Rapson, D. S. (2018). Regression discontinuity in time: Considerations for empirical applications. *Annual Review of Resource Economics*, 10, 533–552.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13–39.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
