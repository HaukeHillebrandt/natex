> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/lmarena-sycophancy-aba/) · [PDF in this repo](./main.pdf)

# Introduction

In late April 2025 OpenAI shipped an update to GPT-4o that made ChatGPT, in the company’s own words, “overly flattering or agreeable — often described as sycophantic,” and rolled it back within days . The two postmortems attribute the failure partly to reward signals built from user feedback — exactly the mechanism the academic literature had predicted: show that human raters and preference models sometimes prefer convincingly written sycophantic responses over correct ones, so that optimizing against human preference judgments can *produce* sycophancy, and document high rates of “social sycophancy” across deployed models. What that literature measures with one-off cross-model snapshots, the April episode supplies as a rare *longitudinal* event: a dated behavior change and a dated reversal.

This paper asks the money question implied by the mechanism: did the sycophantic build actually *win votes*? The natural laboratory is LMArena (Chatbot Arena), the crowdsourced pairwise-battle platform of , whose published battle stream brackets the episode. If human preferences reward sycophancy at the margin, the treated model’s opponent-adjusted win propensity — its Bradley–Terry strength — should rise during the deployment window and fall back after the rollback. The episode has the structure of an interrupted time series with reversal, whose inferential logic — segmented regression at pre-declared cut dates, plus controls that cannot have been treated — is standard in the policy-evaluation literature . Pinned dated snapshots serve as the placebo arm: their weights cannot change, so any step they exhibit at a cut date measures platform-side confounding. Both halves of the design earn their keep here, one by working — a precise null on the vote-winning mechanism — and one by failing loudly: a pinned control moves, mirror-imaged, on exactly the rollback date, converting the “reversal” into an identified arena-side artifact. All estimation follows the open-source `natex` toolkit’s honest-inference conventions ; the automated `natex` survey of the same panel is reported as run, refusals included.

# Data

The source is the public `arena-human-preference-140k` release by LMArena (seven parquet files plus battle metadata, sha256-verified): 135,634 battles from 2025-04-17 00:20 to 2025-07-24 23:59 covering 53 models, with full response text, per-battle nanosecond timestamps, language and category tags, and per-side conversation metadata. Sanity: 0 of 135,634 battle ids show a parquet-vs-metadata timestamp mismatch; per-model side-A vs. side-B mean-token correlation is $`0.992`$ with target side-A share $`0.499`$ (no role-assignment artifact); 0 of 7,650 target responses are zero-token.

The treated series is `chatgpt-4o-latest``-20250326` (“4o”): 7,650 battles — 708 pre-window, 445 deploy-window, 1,374 in the three weeks after rollback. Controls are nine pinned dated snapshots present throughout, led by `gpt-4.1-2025-04-14`, `claude-3-7-sonnet-20250219`, and dated Gemini preview builds. Outcomes are (i) computable style metrics from conversation metadata — bold markers, headers, and list items per 1,000 response tokens, response tokens, turns — and (ii) win/loss/tie votes. Following the pre-registered design (Design 1 of the source corpus’s `DESIGNS.md`), raw win rate is never the estimand: opponent mix and sampling weights move across the episode (4o battles/day: $`88.5 \to 111.3 \to 75.1`$), so votes enter only through windowed Bradley–Terry models.

# Design and Methods

#### ABA interrupted time series.

Two pre-declared cuts — deploy 2025-04-25, rollback 2025-04-28/29 — give an 8–4–8-day template (A1 Apr 17–24; B Apr 25–28; A2 Apr 29–May 6); because the break appears in-data on Apr 26 (staged rollout), the deploy leg is also estimated behavior-dated (8-day-pre/3-day-post). Segment-mean OLS uses CR1 errors clustered by date; adjusted specifications condition on language, category flags, code, weekend, and log turns; a DiD variant subtracts the pooled pinned-control change.

#### Permutation inference.

Every headline statistic is benchmarked against a sliding-cut placebo distribution: the identical window template is recomputed at every feasible alternative cut date in the 99-day panel (77 placebo cuts for the deploy dip, 72 for the rollback step) and the two-sided rank of the true cut is reported; the minimum attainable $`p`$ is $`0.013`$–$`0.014`$, and headline $`p`$-values sit at that floor. A model-placebo grid recomputes each statistic for every sufficient-volume model (17 at the deploy cut, 19 at rollback).

#### Bradley–Terry with segment-specific strength.

All decided battles among the window’s models (Apr 17–May 6; decided 4o battles per segment $`536/342/434`$) enter a logistic MLE in which every model has one strength and 4o has a separate strength per segment ; the deploy jump is $`b_B - b_{A1}`$. The same machinery yields per-model BT steps at the rollback date for the placebo grid.

#### Treatment self-diagnosis.

Per the pre-registered design, whether LMArena’s 4o alias tracked OpenAI’s live model was unresolved but self-diagnosing: no discontinuity at the cuts would mean the arena copy was pinned and 4o becomes a control — informative either way. Resolved here: a deploy-timed discontinuity is present, so 4o was live and is treated.

#### Automated survey, stated as run.

The `natex` survey pipeline ran on the 45-model $`\times`$ 99-day panel (seed 0, $`q=99`$): RDD null (scan $`p=1.00`$); DiD null (scan $`p=0.42`$, and its auto-configuration chose the outcome `n_battles` — volume, which the design spec forbids interpreting); the self-configured synthetic control failed ($`t_0=0`$, no pre-period — recorded, not patched); kink/IV/bunching needs-input; DEE skipped. A declared-input synthetic control (`natex` donors, decided win rate, $`t_0=`$ day 12) gives ATT(post) $`=+0.233`$, pre-RMSPE $`0.056`$, post $`0.249`$, ratio $`4.44`$, in-space placebo $`p=0.067`$ (minimum attainable $`1/15`$) — but its top donor (weight $`0.44`$) is the contaminated `claude-3-7` snapshot, so it is descriptive only.

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" style="width:90.0%" />
<figcaption>(a) Daily mean bold markers per 1<span>,</span>000 response tokens, 2025-04-17 to 2025-06-30: the live 4o alias (blue) dips during the deploy window (shaded) and steps up persistently on 29 April; the pinned <code>claude-3-7</code> snapshot (orange) mirror-images that step on exactly the same date although its weights cannot change; eight other pinned snapshots pooled (gray) are flat. (b) 4o Bradley–Terry strength by segment (95% intervals): the deploy jump is <span class="math inline">+0.02</span> log-odds (SE <span class="math inline">0.14</span>) — no opponent-adjusted votes won; the open 29-April marker is contaminated by the same-day arena-side change.</figcaption>
</figure>

#### Deploy leg: a credible, 4o-specific style break …

Behavior-dated at 2025-04-26 (8-day-pre/3-day-post), 4o’s markdown collapses: headers per 1k tokens fall from $`3.52`$ to $`1.12`$ (dip $`-2.406`$, sliding-cut permutation $`p=0.0128`$, rank 1/78, placebo $`|q_{90}|=0.52`$; Welch $`t=-9.85`$, $`p=6.1\times 10^{-22}`$) and bold falls from $`11.90`$ to $`8.22`$ (dip $`-3.683`$, $`p=0.0769`$). Pooled pinned controls are flat (header dip $`-0.18`$, bold $`-1.51`$). The segmented ABA jump at the Apr-25 cut is $`-1.831`$ headers/1k (CR1 SE $`0.526`$), robust to language/category/code/weekend/turns adjustment ($`-1.723`$, SE $`0.446`$); bold $`-2.965`$ (SE $`1.285`$), adjusted $`-2.749`$ (SE $`0.915`$); the DiD against pooled pinned controls gives $`-1.676`$ (SE $`0.539`$). In the model-placebo grid at the true cuts, 4o’s transient dip ranks 1/17 on both bold and header. The break is real, 4o-specific, and correctly timed — the arena alias was not pinned.

#### … and a clean null on the vote mechanism.

The segment-specific BT deploy jump is $`+0.021`$ log-odds (SE $`0.144`$): implied win rate against the fixed A1 opponent pool moves $`0.540 \to 0.545`$. During the four days users saw the sycophantic build, it gained *zero* opponent-adjusted win rate.

#### Rollback leg: a large step that is not a reversal.

At 2025-04-29 (8-day/8-day windows) 4o steps up, not back: bold $`+12.110`$/1k (permutation $`p=0.0137`$, rank 1/73, placebo max $`|3.70|`$), headers $`+4.246`$ ($`p=0.0137`$, placebo max $`|0.62|`$), response tokens $`+107.8`$ ($`p=0.123`$), BT $`+0.537`$ log-odds (SE $`0.134`$; $`p=0.0137`$, placebo max $`|0.31|`$). The new regime persists through 2025-07 (monthly bold $`21`$–$`22`$/1k vs. $`12.2`$ pre): no reversal to baseline ever occurs. The episode is A–B–C, not A–B–A.

#### The decisive diagnostic: a pinned control moves.

The pinned `claude-3-7` snapshot (full name in Section <a href="#sec:data" data-reference-type="ref" data-reference="sec:data">2</a>) steps on exactly 2025-04-29, mirror-imaged and persistent: bold $`-5.01`$/1k, headers $`-4.63`$/1k, response tokens $`-214`$, BT $`-0.539`$ log-odds (SE $`0.187`$). Pinned weights cannot change, so LMArena changed its serving configuration that day; arena-wide prompt mix was stable (battles/day $`578 \to 557`$, English share $`0.639 \to 0.672`$). Every 29-April estimate — including 4o’s — is therefore contaminated. In the rollback placebo grid 4o ranks 1/19 on bold and 2/19 on header, behind exactly this contaminated control. The raw decided win rate ($`0.549 \to 0.529 \to 0.675`$; full-post $`0.663`$) loads entirely on the confounded boundary and never reverts: the widely repeated “sycophancy raised the win rate, rollback lowered it” narrative is an artifact of the arena change.

<table id="tab:main">
<caption>Headline estimates. Permutation <span class="math inline"><em>p</em></span>-values are two-sided sliding-cut ranks; <span class="math inline">0.013</span>–<span class="math inline">0.014</span> is the minimum attainable.</caption>
<thead>
<tr>
<th style="text-align: left;">Quantity</th>
<th style="text-align: right;">Estimate</th>
<th style="text-align: right;">SE</th>
<th style="text-align: right;">perm. <span class="math inline"><em>p</em></span></th>
<th style="text-align: left;">placebo benchmark</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: deploy leg (cut 2025-04-25/26), 4o</em></td>
</tr>
<tr>
<td style="text-align: left;">header/1k dip (8d/3d)</td>
<td style="text-align: right;"><span class="math inline">−2.406</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.0128</span></td>
<td style="text-align: left;"><span class="math inline">|<em>q</em><sub>90</sub>| = 0.52</span>; rank 1/78</td>
</tr>
<tr>
<td style="text-align: left;">bold/1k dip (8d/3d)</td>
<td style="text-align: right;"><span class="math inline">−3.683</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.0769</span></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">header ABA jump (CR1)</td>
<td style="text-align: right;"><span class="math inline">−1.831</span></td>
<td style="text-align: right;"><span class="math inline">0.526</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">controls <span class="math inline">−0.18</span></td>
</tr>
<tr>
<td style="text-align: left;">adjusted</td>
<td style="text-align: right;"><span class="math inline">−1.723</span></td>
<td style="text-align: right;"><span class="math inline">0.446</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">header DiD vs. controls</td>
<td style="text-align: right;"><span class="math inline">−1.676</span></td>
<td style="text-align: right;"><span class="math inline">0.539</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">BT jump (log-odds)</td>
<td style="text-align: right;"><span class="math inline">+0.021</span></td>
<td style="text-align: right;"><span class="math inline">0.144</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"><em>null: no votes won</em></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: rollback date (2025-04-29, 8d/8d steps), 4o</em></td>
</tr>
<tr>
<td style="text-align: left;">bold/1k step</td>
<td style="text-align: right;"><span class="math inline">+12.110</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.0137</span></td>
<td style="text-align: left;">max <span class="math inline">|3.70|</span>; rank 1/73</td>
</tr>
<tr>
<td style="text-align: left;">header/1k step</td>
<td style="text-align: right;"><span class="math inline">+4.246</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.0137</span></td>
<td style="text-align: left;">max <span class="math inline">|0.62|</span></td>
</tr>
<tr>
<td style="text-align: left;">response tokens step</td>
<td style="text-align: right;"><span class="math inline">+107.8</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;"><span class="math inline">0.123</span></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">BT step (log-odds)</td>
<td style="text-align: right;"><span class="math inline">+0.537</span></td>
<td style="text-align: right;"><span class="math inline">0.134</span></td>
<td style="text-align: right;"><span class="math inline">0.0137</span></td>
<td style="text-align: left;">max <span class="math inline">|0.31|</span></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: pinned <code>claude-3-7-sonnet-20250219</code>, same date (violation)</em></td>
</tr>
<tr>
<td style="text-align: left;">bold/1k step</td>
<td style="text-align: right;"><span class="math inline">−5.01</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">weights cannot change</td>
</tr>
<tr>
<td style="text-align: left;">header/1k step</td>
<td style="text-align: right;"><span class="math inline">−4.63</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">response tokens step</td>
<td style="text-align: right;"><span class="math inline">−214</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">BT step (log-odds)</td>
<td style="text-align: right;"><span class="math inline">−0.539</span></td>
<td style="text-align: right;"><span class="math inline">0.187</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>

Notes: style metrics are per 1,000 response tokens. Dips use behavior-dated 8-day-pre/3-day-post windows; steps use 8-day/8-day windows at the cut. CR1 clusters by date. BT $`=`$ segment- or window-specific Bradley–Terry strength over all decided battles among window models.

# Caveats and Conclusion

Four caveats bound the claims. First, per the pre-registered design, the style metrics computable from conversation metadata are a *proxy* for sycophancy — directional evidence of a behavior change, not a sycophancy measurement; LLM-judge scoring of the stored response text is the designated follow-up. The conclusion that survives this caveat is exactly the null: whatever the treated build did to users, it did not win opponent-adjusted votes. Second, the rollback leg is unidentified: the arena-side serving change of 2025-04-29 sits on the same date, so no 29-April estimate separates OpenAI’s rollback from LMArena’s change. Third, permutation $`p`$-values are floor-limited ($`1/78`$, $`1/73`$) by the panel length; they establish rank-1 extremity, not fine-grained significance. Fourth, the declared-input synthetic control is descriptive only (its top donor is the contaminated control), and the survey’s self-configured families returned nulls or refusals, reported above as run.

The conclusion is a split verdict that the design itself delivered. The deploy leg shows the ABA machinery working: a sharp, correctly timed, 4o-specific break, flat pinned controls, and a precise behavioral null — sycophancy did not win votes. The rollback leg shows the machinery catching its own confound: a pinned model that cannot change, changing — the single observation that converts the entire post-rollback narrative, including the celebrated raw win-rate rise, into an arena-side artifact. Platform-side serving changes are an underappreciated threat for any study that treats arena time series as model behavior; pinned snapshots are the cheap, decisive placebo that catches them.

#### Reproducibility.

All estimates derive deterministically (seed 0) from the public battle release; no dataset files are committed. Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates from the committed `figures/make_fig.py`, which asserts the headline estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Lopez Bernal, J., Cummins, S., and Gasparrini, A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 46(1), 348–355.

Bradley, R. A., and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345.

Cheng, M., Yu, S., Lee, C., Khadpe, P., Ibrahim, L., and Jurafsky, D. (2025). Social sycophancy: A broader understanding of LLM sycophancy. arXiv:2505.13995.

Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhu, B., Zhang, H., Jordan, M., Gonzalez, J. E., and Stoica, I. (2024). Chatbot Arena: An open platform for evaluating LLMs by human preference. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:8359–8388.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation*. Software. <https://github.com/HaukeHillebrandt/natex>

OpenAI (2025). Sycophancy in GPT-4o: What happened and what we’re doing about it. Blog post, 29 April 2025. <https://openai.com/index/sycophancy-in-gpt-4o/>

OpenAI (2025). Expanding on what we missed with sycophancy. Blog post, 2 May 2025. <https://openai.com/index/expanding-on-sycophancy/>

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., and Perez, E. (2024). Towards understanding sycophancy in language models. In *International Conference on Learning Representations (ICLR)*.

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
