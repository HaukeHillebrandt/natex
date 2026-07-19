> **Markdown render for GitHub browsing** — typeset versions: [HTML](https://haukehillebrandt.github.io/natex/export-controls-three-leg/) · [PDF](https://haukehillebrandt.github.io/natex/export-controls-three-leg.pdf) · [PDF in this repo](./main.pdf)

# Introduction

On 7 October 2022 the US Bureau of Industry and Security imposed the first broad export controls on advanced AI accelerators and chipmaking tools bound for China, a policy contemporaries immediately read as a landmark attempt to choke off China’s access to frontier AI compute . The rule’s performance-and-interconnect thresholds left room for compliant workarounds — Nvidia’s A800 and H800 — and on 17 October 2023 a second round closed that loophole, banning the workaround chips outright. Whether such controls work is an old question in trade policy: export controls impose costs on the controlling country’s own firms and reroute rather than stop trade when substitutes exist . Firm-level evidence from the earlier entity-list controls finds large collateral costs — US suppliers to targeted Chinese firms lose customers, market value (roughly \$130 billion), lending, and employment, with broad supply-chain decoupling but no reshoring . On the evasion margin, warned in 2023 that AI-chip smuggling into China, then likely in the hundreds to low thousands of units, would industrialize; by 2026, diversion-and-resale estimates put smuggled compute at hundreds of thousands of H100-equivalents . What this literature lacks is a quasi-experimental estimate of whether the controls actually bent China’s compute accumulation and large-scale training activity.

This paper provides three complementary tests — three legs of one story — using two Epoch AI datasets and the `natex` toolkit . Leg 1 is a country-level difference-in-differences (DiD) on the quarterly count of $`\geq 10^{24}`$-FLOP training runs, with a scan-based localization step . Leg 2 is a group difference-in-kinks (DiK) — the design of , building on the regression kink design of — contrasting the growth-rate bend in China’s legal chip stock against US hyperscalers at the 2023-10-17 rule, with placebo cutoffs in the spirit of . Leg 3 repeats the DiK on China’s total stock including smuggled chips and quantifies the smuggling substitution directly. The verdict is deliberately mixed and stated as such: Leg 2 is credible, Leg 3 is attenuated and fragile, Leg 1 is descriptive only.

# Data

#### Country panel (Leg 1).

From Epoch AI’s notable-models database we build a panel of 4 country groups (US, China, EU, Other) $`\times`$ 26 quarters (2020Q1–2026Q2), $`n=104`$ cells. The outcome is the count of training runs $`\geq 10^{24}`$ FLOP per quarter; a secondary outcome is the median $`\log_{10}`$ training compute. Models are assigned to countries by the lead (first-listed) organization, so multinational collaborations are attributed to the lead country. The declared treatment is `china_controlled`: China from 2022Q4 (the first control round) onward. Two caveats bind: 2026 quarters undercount compute-known runs ($`\sim`$<!-- -->13% FLOP coverage, a reporting lag), and counts measure quantity of big runs, not frontier capability.

#### Chip stocks (Legs 2–3).

The chip-stock series come from Epoch AI’s AI Chip Owners data : quarterly cumulative H100-equivalent (H100e) stock by designer. From these we form ln cumulative stock series for: China’s legal stock; China’s total stock including the smuggled Nvidia series ; the four US hyperscalers (Amazon, Microsoft, Google, Meta) as the primary control; neoclouds (Oracle, CoreWeave, xAI) as an alternative control; and an “Other” group used as a placebo-treated group. Each series covers 16 quarters from 2022-03-31; the incomplete 2026-03-31 quarter is dropped (8 rows whose “cumulative” totals decrease). The running variable is days since 2023-10-17. The smuggled series begins 2024-03-31 and is itself part of the treatment response — which is exactly why Legs 2 and 3 are reported side-by-side.

# Design and Methods

#### Leg 1: SuDDDS DiD.

The `natex` sector-by-time DiD scan (SuDDDS) searches subsets of units and break dates maximizing a likelihood-ratio scan statistic , calibrated by permutation ($`q=99`$, seed 7, windows $`\{4,7,13\}`$). We run it three ways: (i) known-treatment mode (Bernoulli model, `wcc`) on the declared treatment; (ii) a normal-model cross-check with a dependence-preserving `ar1_unit` null; (iii) discovery mode with no treatment column, asking whether the export-control dates would be found autonomously. Effects for China at 2022Q4 come from DD, synthetic-control, and GESS estimators, followed by a manual placebo battery (placebo-in-space with the Abadie china-removed convention, placebo-in-time at 2021Q3).

#### Legs 2–3: group difference-in-kinks.

In the DiK contrasts the same schedule kink across two *time periods*. Here the export-license schedule bends *for China only* at a common date, so the Böckerman contrast is taken across *groups* at one cutoff: $`\mathrm{DiK} = (\text{China slope kink}) - (\text{hyperscaler slope kink})`$, each kink estimated by local linear regression on each side of the cutoff (right-minus-left), triangular kernel, primary bandwidth 548 days. Identification is parallel-kink: absent the controls, China’s growth bend at Oct-2023 would have matched the hyperscalers’; the control group differences out the global supply-curve bend (the hyperscalers’ own kink is $`-0.000666`$/day). Robustness: bandwidths $`\{365,548,730\}`$ days $`\times`$ $`\{`$triangular, uniform$`\}`$ kernels, 45-day donuts, six placebo cutoffs (including the Oct-2022 first round at $`-375`$ days — the A800/H800 loophole-round anticipation contrast), and a placebo-treated group (Other vs hyperscalers, both untreated), in the spirit of . Leg 3 re-runs the identical design on China’s total stock including smuggled chips.

#### Honest-inference notes, stated as run.

The pipeline’s refusals and trivial passes are reported, not patched over. Leg 1: the built-in `tau_randomization_test` correctly refuses ($`p=\mathrm{NaN}`$, 3 non-treated units $`<5`$ usable placebos) rather than fabricating a $`p`$-value, so the manual placebo-in-space battery is the inference of record; the scan $`p=0.010`$ is the $`q=99`$ permutation floor; with 4 units, all panel $`p`$-values are descriptive. Leg 2: HC1 standard errors on 6 quarterly points per cell overstate precision on a cumulative-stock series; CR1 clustering by calendar year keeps significance but has $`G=4`$ clusters with $`t(G-1)`$ critical values — a known few-cluster over-rejection regime ; the sign-stability across 25 specifications is the real evidence. The density-kink-difference test passes trivially (both groups share an identical quarterly grid; estimate exactly $`0.0`$, $`p`$ reported as null), no predetermined covariates exist in a two-group quarterly aggregate (the placebo-treated group and the Oct-2022 placebo round substitute for the covariate-kink battery), and no Fieller set is produced (sharp design, known unit denominator). A recorded `natex` UX finding: the CLI aliases the group contrast through its time interface (`--time china --t0 1`), so output cells are labeled `pre_*`/`post_*` and its caveat text misdescribes the cross-group contrast as a “time-stable marginal response.”

# Results

<figure id="fig:fig1" data-latex-placement="t">
<img src="fig1" />
<figcaption>(a) Leg 1: quarterly count of <span class="math inline"> ≥ 10<sup>24</sup></span>-FLOP training runs by country group, with both export-control rounds marked; the shaded region is the <span class="math inline">∼</span>13%-coverage reporting-lag zone. The panel is a two-speed world: US and China accelerate, EU and Other stagnate. (b) Legs 2–3: ln cumulative H100e chip stock for the hyperscaler control group, China’s legal stock, and China’s total stock including smuggled chips, around the 2023-10-17 cutoff. (c) Leg 2: group-DiK estimates (bandwidth 548 days, triangular kernel) at the true cutoff (diamond) and at six placebo cutoff placements, with 95% HC1 intervals; the Oct-2022 loophole round at <span class="math inline">−375</span> days is null.</figcaption>
</figure>

#### Leg 1: localization succeeds, attribution fails.

In known-treatment mode SuDDDS recovers the declared treatment exactly: top subset $`\{`$China$`\}`$, $`T_0=11=`$ 2022Q4, window $`W=13`$, Bernoulli LLR $`6.3715`$, panel-null $`p=0.010`$ (the $`q=99`$ floor); the composition check passes ($`p=1.0`$) and the anticipation test passes (Holm $`p=1.0`$). The normal-model cross-check under the dependence-preserving `ar1_unit` null finds the mirror image — the complement $`\{`$EU, Other, US$`\}`$ at $`T_0=13`$ (2023Q2), LLR $`14.9487`$ — but $`p=0.14`$: a 4-unit panel cannot separate a persistent step from unit-level AR(1). Discovery mode does *not* find the controls: the top hit is a $`\{`$China, US$`\}`$ *downshift* at 2025Q4 (LLR $`11.277`$, $`p=0.13`$; runner-ups ALL@2025Q4, LLR $`7.232`$, and $`\{`$China, US$`\}`$@2024Q2, LLR $`5.835`$), with no export-control date in the top five — and the 2025Q4 downshift is at least partly the reporting-lag artifact. Effect estimates for China at 2022Q4 (Table <a href="#tab:main" data-reference-type="ref" data-reference="tab:main">1</a>, panel A) are uniformly *positive*: DD $`\hat\tau=+3.214`$ runs/qtr (se $`1.157`$, $`t=2.779`$, 95% CI $`[+0.947,
+5.481]`$), synthetic control $`+3.933`$, GESS $`+3.842`$, DD at the 2023Q4 round $`+4.455`$, and DD on median $`\log_{10}`$ compute $`+0.649`$ $`[+0.322, +0.976]`$. Taken literally: no suppression — China’s big-run count *outgrew* every control-implied counterfactual. But the placebo battery kills any causal reading of the magnitude: pseudo-treating each control unit at the same date yields studentized $`|t|`$ of $`5.37`$ (EU), $`2.86`$ (Other), and $`5.16`$ (US) against China’s $`2.78`$ (china-removed convention) — exact placebo-in-space $`p = 4/4 = 1.0`$. Placebo-in-time is clean (China at 2021Q3: $`\tau=-0.30`$, $`t=-1.33`$), and the built-in randomization test correctly refused ($`p=\mathrm{NaN}`$, $`3<5`$ placebos). The robust statement is qualitative only: China’s large-run count did not fall behind its counterfactual after either control round.

#### Leg 2: the legal channel credibly bent.

The headline group DiK (China legal vs hyperscalers, 2023-10-17, bandwidth 548 days, triangular kernel) is $`-0.0015404`$ ln-units/day (HC1 se $`0.000467`$, $`t=-3.298`$, 95% CI $`[-0.002456, -0.000625]`$), i.e. $`\approx -0.56`$ ln-units/yr of growth-rate change; with CR1 clustering by year, se $`0.000299`$, $`t=-5.152`$ ($`G=4`$, few-cluster caveat). The cell kinks are $`-0.000666`$/day (hyperscalers) vs $`-0.002206`$/day (China legal). The estimate is negative in **25 of 25** estimated specifications: $`-0.000869`$ to $`-0.002026`$ across bw365/548/730 $`\times`$ triangular/uniform ($`t`$ from $`-2.07`$ to $`-6.57`$), and $`-0.0018996`$/$`-0.0012108`$/$`-0.0008011`$ on the 45-day-donut grid ($`t`$ $`-3.92`$/$`-2.04`$/$`-1.60`$). The placebo-cutoff battery (bw548; empirical size 2/6) localizes the kink to within one quarter of the rule: the Oct-2022 first round at $`-375`$ days is *null* at $`+0.0018203`$ (se $`0.001251`$, $`p=0.146`$) — the loophole round produced no bend, a pseudo-pre-trend pass — and $`-183`$d ($`p=0.578`$), $`+183`$d ($`p=0.364`$), $`+275`$d ($`p=0.805`$) are null, while $`-92`$d ($`p=0.0089`$) and $`+92`$d ($`p=0.0240`$) reject: with quarterly data, one-quarter shifts still straddle the same bend, which still brackets 2023-10-17. The placebo-treated group (Other vs hyperscalers, both untreated) is null at bw548/730 ($`t=+1.34`$, $`-0.29`$, $`-0.20`$, $`-0.75`$) but rejects at bw365 (triangular $`+0.000958`$, $`t=2.49`$; uniform $`+0.001219`$, $`t=3.04`$) — parallel-kink fails among untreated groups at four-points-per-cell bandwidths, so the bw365 cells are discounted and the defensible band is bw548–730: $`-0.0009`$ to $`-0.0015`$/day ($`\approx -0.3`$ to $`-0.56`$ ln-units/yr).

#### Leg 3: smuggling halves and de-robustifies the effect.

On China’s total stock including smuggled chips, the same design gives $`-0.0007638`$ (se $`0.000496`$, $`t=-1.540`$), with a 95% CI of $`[-0.001736, +0.000209]`$ that spans zero — roughly half the legal-channel bend, and $`t`$ ranges from $`-0.55`$ to $`-3.08`$ across the six specifications (the only rejection is at the discounted bw365). The mechanism is visible in the raw series: smuggled Nvidia compute in China grew from 27.8k H100e (Q1 2024) to 662k H100e (Q4 2025), doubling every 4.7 months — the fastest-growing series in the dataset — and now stands at $`1.65\times`$ China’s legal Nvidia stock. Even so, China’s share of the world chip stock roughly halved to 8.7% over the period despite absolute growth: the controls did not stop China’s accumulation, but China fell behind a world that accelerated faster.

<table id="tab:main">
<caption>Headline estimates across the three legs.</caption>
<thead>
<tr>
<th style="text-align: left;">Specification</th>
<th style="text-align: right;"><span class="math inline"><em>τ̂</em></span></th>
<th style="text-align: right;">se</th>
<th style="text-align: right;"><span class="math inline"><em>t</em></span></th>
<th style="text-align: left;">note</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel A: Leg 1 — DiD, China at 2022Q4 (runs <span class="math inline"> ≥ 10<sup>24</sup></span> FLOP per quarter)</em></td>
</tr>
<tr>
<td style="text-align: left;">DD (control <span class="math inline">=</span> other 3 units)</td>
<td style="text-align: right;"><span class="math inline">+3.214</span></td>
<td style="text-align: right;"><span class="math inline">1.157</span></td>
<td style="text-align: right;"><span class="math inline">2.78</span></td>
<td style="text-align: left;">CI <span class="math inline">[+0.947, +5.481]</span></td>
</tr>
<tr>
<td style="text-align: left;">Synthetic control</td>
<td style="text-align: right;"><span class="math inline">+3.933</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">CI <span class="math inline">[+1.533, +6.334]</span></td>
</tr>
<tr>
<td style="text-align: left;">GESS</td>
<td style="text-align: right;"><span class="math inline">+3.842</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">CI <span class="math inline">[+1.442, +6.243]</span></td>
</tr>
<tr>
<td style="text-align: left;">DD at 2023Q4 round</td>
<td style="text-align: right;"><span class="math inline">+4.455</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">CI <span class="math inline">[+1.719, +7.190]</span></td>
</tr>
<tr>
<td style="text-align: left;">DD, median <span class="math inline">log<sub>10</sub></span> compute</td>
<td style="text-align: right;"><span class="math inline">+0.649</span></td>
<td style="text-align: right;">—</td>
<td style="text-align: right;">—</td>
<td style="text-align: left;">CI <span class="math inline">[+0.322, +0.976]</span></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;">placebo-in-space exact <span class="math inline"><em>p</em> = 4/4 = 1.0</span> <span class="math inline">⇒</span> descriptive, not causal</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel B: Leg 2 — group DiK, ln H100e per day, China legal vs hyperscalers</em></td>
</tr>
<tr>
<td style="text-align: left;">bw548 triangular (headline)</td>
<td style="text-align: right;"><span class="math inline">−0.001540</span></td>
<td style="text-align: right;"><span class="math inline">0.000467</span></td>
<td style="text-align: right;"><span class="math inline">−3.30</span></td>
<td style="text-align: left;">HC1</td>
</tr>
<tr>
<td style="text-align: left;">bw548 triangular, CR1 by year</td>
<td style="text-align: right;"><span class="math inline">−0.001540</span></td>
<td style="text-align: right;"><span class="math inline">0.000299</span></td>
<td style="text-align: right;"><span class="math inline">−5.15</span></td>
<td style="text-align: left;"><span class="math inline"><em>G</em> = 4</span> clusters</td>
</tr>
<tr>
<td style="text-align: left;">bw548 uniform</td>
<td style="text-align: right;"><span class="math inline">−0.001132</span></td>
<td style="text-align: right;"><span class="math inline">0.000525</span></td>
<td style="text-align: right;"><span class="math inline">−2.16</span></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">bw730 triangular / uniform</td>
<td style="text-align: right;"><span class="math inline">−0.001140</span> / <span class="math inline">−0.000869</span></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"><span class="math inline">−2.85</span> / <span class="math inline">−2.07</span></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">bw365 triangular</td>
<td style="text-align: right;"><span class="math inline">−0.002026</span></td>
<td style="text-align: right;"><span class="math inline">0.000308</span></td>
<td style="text-align: right;"><span class="math inline">−6.57</span></td>
<td style="text-align: left;">discounted</td>
</tr>
<tr>
<td style="text-align: left;">placebo cutoff <span class="math inline">−375</span>d (Oct-2022)</td>
<td style="text-align: right;"><span class="math inline">+0.001820</span></td>
<td style="text-align: right;"><span class="math inline">0.001251</span></td>
<td style="text-align: right;"></td>
<td style="text-align: left;"><span class="math inline"><em>p</em> = 0.146</span>: null</td>
</tr>
<tr>
<td colspan="5" style="text-align: left;"><em>Panel C: Leg 3 — group DiK, China total stock (incl. smuggled) vs hyperscalers</em></td>
</tr>
<tr>
<td style="text-align: left;">bw548 triangular</td>
<td style="text-align: right;"><span class="math inline">−0.000764</span></td>
<td style="text-align: right;"><span class="math inline">0.000496</span></td>
<td style="text-align: right;"><span class="math inline">−1.54</span></td>
<td style="text-align: left;">CI <span class="math inline">[−0.001736, +0.000209]</span></td>
</tr>
<tr>
<td style="text-align: left;">bw730 triangular / uniform</td>
<td style="text-align: right;"><span class="math inline">−0.000458</span> / <span class="math inline">−0.000235</span></td>
<td style="text-align: right;"></td>
<td style="text-align: right;"><span class="math inline">−1.10</span> / <span class="math inline">−0.55</span></td>
<td style="text-align: left;"></td>
</tr>
<tr>
<td style="text-align: left;">bw365 triangular</td>
<td style="text-align: right;"><span class="math inline">−0.001049</span></td>
<td style="text-align: right;"><span class="math inline">0.000340</span></td>
<td style="text-align: right;"><span class="math inline">−3.08</span></td>
<td style="text-align: left;">discounted</td>
</tr>
</tbody>
</table>

Notes: Panel A: `natex` DiD effect estimators on the 4-country $`\times`$ 26-quarter panel (seed 7); 95% CIs in the note column. Panels B–C: sharp group difference-in-kinks, right-minus-left slope contrasts in ln cumulative H100e stock per day at the 2023-10-17 cutoff; HC1 standard errors unless noted; bandwidths in days; “discounted” marks bw365 cells where the placebo-treated group (Other vs hyperscalers) rejects. Multiply DiK estimates by 365.25 for annual growth-rate changes.

# Caveats and Conclusion

Four caveats bound the claims. First, Leg 1’s panel has four units — the floor for any panel placebo inference — and its placebo-in-space failure is informative precisely because it shows the units are not exchangeable (a two-speed world), so no DiD counterfactual built from EU/Other/US is credible for China’s magnitude; every Leg-1 $`p`$-value is descriptive. Second, Leg 2’s inference rests on 6 quarterly points per cell: HC1 overstates precision on a cumulative series, CR1 has $`G=4`$ clusters , and the $`\pm 92`$-day placebo rejections show the kink is localized only to $`\pm 1`$ quarter — the design cannot distinguish 2023-10-17 from an adjacent-week event, though no rival same-quarter candidate of comparable relevance is known to us. Third, the smuggled series underlying Leg 3 is itself an estimate with wide uncertainty and begins only in Q1 2024, endogenously to the treatment. Fourth, counts of big training runs measure quantity, not frontier capability; controls aimed at chip access can bind on quality and efficiency margins invisible to Leg 1’s outcome.

The three-leg verdict: *credible* on the legal channel (Leg 2) — the October-2023 controls bent China’s legal chip-stock growth down by roughly $`0.3`$–$`0.56`$ ln-units/yr relative to every control group, with the loophole-round placebo null exactly where the policy history says it should be; *attenuated and fragile* on total compute (Leg 3) — smuggling substitution roughly halves the point estimate and removes its robustness; and *descriptive only* on training activity (Leg 1) — China’s count of large training runs did not fall behind any counterfactual, but the design cannot certify that as a causal null. Export controls, on this evidence, bent the channel they legally govern, were substantially bypassed in aggregate, and left China’s large-run training activity unbowed.

#### Reproducibility.

All estimates were produced with `natex` v0.2.0 : the Leg-1 runs at seed 7 ($`q=99`$) and the Leg-2/3 battery at seed 20260716 (the DiK estimator itself is deterministic), from frozen extracts of the Epoch AI datasets (not committed to the repository). Figure <a href="#fig:fig1" data-reference-type="ref" data-reference="fig:fig1">1</a> regenerates deterministically from the committed `figures/make_fig.py`, which asserts the headline DD and DiK estimates against the numbers of record before drawing.

<div class="thebibliography">

9

Allen, G. C. (2022). Choking off China’s access to the future of AI. Center for Strategic and International Studies (CSIS), October 2022. <https://www.csis.org/analysis/choking-chinas-access-future-ai>

Böckerman, P., Jysmä, S., and Kanninen, O. (2025). *Difference-in-Kinks Design*. IZA Discussion Paper No. 18313. <https://docs.iza.org/dp18313.pdf>

Bown, C. P. (2020). Export controls: America’s other national security threat. Peterson Institute for International Economics Working Paper 20-8. <https://www.piie.com/sites/default/files/documents/wp20-8.pdf>

Cameron, A. C., and Miller, D. L. (2015). A practitioner’s guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Card, D., Lee, D. S., Pei, Z., and Weber, A. (2015). Inference on causal effects in a generalized regression kink design. *Econometrica*, 83(6), 2453–2483.

Crosignani, M., Han, L., Macchiavelli, M., and Silva, A. F. (2025). Securing technological leadership? The cost of export controls on firms. Federal Reserve Bank of New York Staff Report No. 1096 (first circulated 2023 as “Geopolitical Risk and Decoupling: Evidence from U.S. Export Controls”).

Epoch AI (2026). Data on AI chip owners. <https://epoch.ai/data/ai-chip-owners> (extract of 2026).

Epoch AI (2026). Data on AI models. <https://epoch.ai/data/ai-models> (extract of 2026).

Fist, T., and Grunewald, E. (2023). *Preventing AI Chip Smuggling to China: A Working Paper*. Center for a New American Security (CNAS), October 2023.

Ganong, P., and Jäger, S. (2018). A permutation test for the regression kink design. *Journal of the American Statistical Association*, 113(522), 494–504.

Herlands, W., McFowland III, E., Wilson, A. G., and Neill, D. B. (2018). Automated local regression discontinuity design discovery. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’18)*.

Juniewicz, I. (2026). Diversion and resale: Estimating compute smuggling to China. Epoch AI, April 2026. <https://epoch.ai/publications/chip-smuggling>

Miller, C. (2022). *Chip War: The Fight for the World’s Most Critical Technology*. Scribner, New York.

Hillebrandt, H. (2026). *natex: automated natural-experiment discovery and estimation* (version 0.2.0). Software. <https://github.com/HaukeHillebrandt/natex>

</div>

[^1]: University College London. Email: `ucjthhi@ucl.ac.uk`. This paper and the underlying `natex` software were prepared with substantial assistance from Anthropic’s Claude models; the author reviewed the analyses and text and is responsible for all remaining errors.
