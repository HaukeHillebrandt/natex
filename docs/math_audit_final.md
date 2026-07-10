# Final reconciled math audit — LoRD3, SuDDDS, DEE

**Date:** 2026-07-10.
**Process:** (1) Claude Fable 5 multi-agent audit: 8 derivation agents re-derived all 142 numbered
equations/algorithms from the PDFs; 130 skeptic agents adversarially verified every suspected
issue and every high/medium improvement (`math_audit.md`, raw data `math_audit_raw.json`).
(2) Independent second opinion by GPT 5.6 Sol via Codex, re-deriving section A from the rendered
PDFs and hunting for missed issues (`codex_second_opinion.md`). (3) Disagreements between the two
model families adjudicated by rendering the disputed pages at 576 dpi and reading the actual
typography (this file).

## 1. Adjudicated disagreements (visual inspection, 576 dpi)

| Item | Claude audit said | Codex said | Page says (adjudicated) |
|---|---|---|---|
| `def-ci-bi-sufficient-stats` (thesis p.108/PDF 128) | typo: prints σᵢ, needs σᵢ² | prints σᵢ² — audit misread | **Codex right.** Page clearly prints cᵢ = Σ rᵢ/σᵢ², bᵢ = Σ 1/σᵢ². RETRACTED from errata. |
| `eq6.8-aggregates` (same page) | typo (inherits σ claim) | correct as printed | **Codex right.** Eq 6.8 is correct. RETRACTED from errata. |
| `kdd-sec3.1-poly-exponent` "thesis fixes x^R" | thesis prints x^r | thesis repeats x^R on PDF pp.102 & 127 | **Codex right.** Both thesis pages print γ_r x^R. Typo present in KDD *and* thesis ch.5 *and* ch.6. |

**Root cause of the two retractions:** pypdf text extraction drops superscripts; derivation agent
and two skeptics shared the same extraction path, so the "independent" confirmations were
correlated. Lesson (now in `.claude/napkin.md`): visually render disputed formulas before ruling
on typography. The corrected math everyone implemented is identical either way (σ² weights) — the
error was in attributing a typo to the thesis, not in the math itself.

Everything else in `math_audit.md` section A stands, with Codex's PARTIALLY refinements noted
in its §1 table (which softens ~14 items without overturning them — read both files together).

## 2. Consolidated errata that natex must handle (post-reconciliation)

### Statistical validity (design-level)
1. **Randomization test is a parametric bootstrap, not exact** (both audits agree; Codex adds
   that the code *does* refit f per replica, contra the Claude audit's detail). Repair: honest
   discovery/estimation splitting + correctly stated fitted-null Monte Carlo p-values with +1
   rank correction; don't claim exactness.
2. **Bernoulli null replicas are not Bernoulli(p̂)** (Codex new #5): the released generator draws
   T* = 1{p̂ + σ̂Z > U}, which has success probability E[clip(p̂+σ̂Z,0,1)] ≠ p̂ (≈.176 vs .1 at
   p̂=.1, σ̂=.3). natex: draw T* ~ Bernoulli(p̂) directly.
3. **Placebo tests as printed mechanically reject valid RDDs** (Codex new #4, subsumes the
   Claude finding): side-mean contrasts are nonzero for the running variable even under a valid
   design. natex: local intercept-continuity placebo (side-specific trends, cutoff intercept
   equality), joint/multiplicity-correct, categorical covariates via distributional tests
   (Codex new #18).
4. **Group-instrument estimator (Eq 5.14/KDD 14) inconsistent**: W = T − μ strips the exogenous
   jump and keeps the endogenous residual; plim τ̂ = τ + Var(u)/(Var(εᵀ)+Var(u)) under the
   paper's own DGP. natex: frozen side-indicator 2SLS (primary), f+μ̂ generated-regressor form
   only with proper first-stage inference.
5. **τ̂ placebo test (ch.6)**: one-sided 95th-percentile rule can't detect negative effects and
   placebo subsets aren't exchangeable with the selected treated subset. natex: two-sided
   studentized statistic, +1 rank p-value, matched subset shapes, stated assumptions; the
   "independence of the two tests" claim replaced by the precise conditional statement.
6. **Projected McCrary density test**: oblique projection can hide (or manufacture) density
   jumps, and search-selected geometry invalidates fixed-cutoff critical values. natex:
   rddensity on signed distance along the *frozen* normal from an honest split, documented as a
   falsification test only.
7. **DEE Theorem 1 / Lemmas 1–2**: finite-N variance formula missing cross terms (Lemma 1);
   Z₁ ≤ E[q] step false (Lemma 2) — repairable via sphere geometry E[q²] ≤ 2ρ² (Codex's cleaner
   route, no extra assumption); implication 3 needs the √2 factor and holds only along
   fixed-shape scaling. Also Var-difference sign inconsistency (Codex #32).
8. **DEE mixture intervals** (Codex new #28, high impact): rural-roads code draws an independent
   model indicator per prediction point instead of one per posterior draw, destroying
   between-model covariance — aggregate CIs can be far too narrow. natex: draw the model label
   once per sample; mixture covariance wΣ_β+(1−w)Σ_τ+w(1−w)(μ_β−μ_τ)(μ_β−μ_τ)ᵀ.
9. **Same-data forest + local-IV correlation** (Codex #39 / audit `dee-gp-noise-classical-se`):
   cross-fit the observational estimator vs the discovery/IV stage, or model the joint
   covariance; never treat classical 2SLS SE² as known independent GP noise.
10. **First-stage relevance is not implied by a high residual LLR** (Codex new #9, #37): check
    the actual first-stage jump (and weak-IV-robust intervals) after validation/repair.

### Algorithmic repairs (ch.6 SuDDDS)
11. **Algorithm 6 never stores the global best (s*, T₀*, W*)** across windows/restarts (Codex
    new #12) — keep an incumbent.
12. **Algorithm 7 evaluates boundary cutoffs with an empty side** (Codex new #13) — require
    minimum precision mass both sides.
13. **Algorithm 8 is deletion-only with undefined empty-slice priorities** (Codex new #14;
    the audit's "non-monotone" claim was refuted — skeptics right): repair = relax dimension j,
    slice over all values, retain incumbent explicitly.
14. **Algorithm 9 line 6: argmin** (both audits agree); initialize control-set MSE = +∞.
15. **Single-Δ statistic must profile μᵢ under H₁ and restrict to windows** (Codex new #16;
    numeric counterexample LLR 4.74 vs correct 5.33): use C̃ᵢ, B̃ᵢ with the precision-weighted
    δ̄ correction; scan both signs of Δ.
16. **WCC ρ-priority scan is heuristic, not exact** (confirmed; Codex adds the useful special
    case: exact ordering when B₁/B₂ ratio is common). natex default at small cardinality:
    exhaustive per-dimension enumeration (2^V−1 subsets), which is exact for the printed
    statistic.
17. **Eq 6.10 prose swaps q₁/q₂ ↔ β_g0/β_g1** (typo confirmed; the equations are right).
18. **Panel replica nulls must preserve unit/time dependence** (Codex new #19) and DiD-in-time
    McCrary on calendar time is information-free (Codex new #17) — replace with composition/
    anticipation checks.
19. **Continuous-treatment estimand** (Codex new #20): DD contrast estimates ζ·τ, not τ; needs
    dose normalization or IV form. **Model class must match T's type** (Codex new #21):
    Bernoulli/Poisson variants needed for binary/count treatments in the DiD scan.

### Implementation landmines (legacy code, affects backtest comparisons)
20. **Reverse-neighbor variance bug** (Codex new #1, high impact): `sigma_2_all[n_i] =
    np.var(r[neighs[n_i]])` indexes a *row* of the neighborhood matrix (centers whose
    neighborhoods contain i, variable size, can be singleton→zero variance→floor weights),
    not point i's own kNN column. Every legacy Normal-model score and its null replicas are
    affected; do NOT treat legacy outputs as ground truth for parity tests of σ̂ᵢ².
21. Sharp-RDD (pure-group) bisections dropped as NA in the R package — natex scores them with
    the boundary likelihood supremum.
22. Bisection bracket-doubling runaway at boundary probabilities; comment's monotonicity claim
    wrong (bisect on βS(β)); complement-dedup dead code (both packages).
23. Hyperplane tie convention: the center point lies on every candidate plane — define
    membership explicitly (natex: signed distance ≥ 0 → group 1, documented).
24. 1e-6 absolute variance floor → scale-dependent; use data-scaled shrinkage (with Codex's
    caveat that the audit's numeric example overstated the ceiling's universality).

### Pure typos (documented in method cards, no design impact)
x^R→x^r (all three occurrences); Eq 5.18 union→intersection; Eq 5.21 needs inverse-logit AND
log μᵢ (Codex sharpened); Eq 6.4 rank deficiency (reference constraints); Eq 6.18/6.20
denominators count summed records + s→s_τ subscript (Codex #22); Eq 6.19 unit-level weights;
Eq 6.21 ill-typed; Eqs 6.24–6.25 dimensionally invalid as printed (Codex #23); Eq 6.27 creates
heteroskedasticity not correlation — the GESS synthetic experiment's stated mechanism is wrong
though the experiment itself still shows *a* misspecification GESS handles (Codex #24);
Eq 6.28 set notation (Codex #25); thesis xref 6.26→5.22; DEE: missing mixture weights in
weighted-BLOOCV sum, M vs M′, σᵢ SD-vs-variance in LOOCV prose (Codex #31), bias sign
convention drift across appendices (Codex #30), Algorithm 2 returns wrong tuple type
(Codex #34), Assumption 4 ill-typed independence statement → conditional mean independence
(Codex #35), Appendix F.2 α_Y→α_B (Codex #36).

## 3. Improvement decisions after both reviews

Adopted for natex (with Codex refinements):
- Unknown-variance GLR per split: k/2·log(RSS₀/RSS₁) (exact monotone-in-t² for common variance);
  keep the heteroskedastic plug-in variant as an option with documented caveats.
- Side-specific **local-linear alternative with slopes under BOTH hypotheses** (Codex caught
  that the audit's version — slopes only under H₁ — rejects on pure kinks; null = shared-
  intercept two-slope model, alternative adds intercept jump).
- Pure-group supremum scoring; bracketed Newton/IRLS in θ=log β with Firth fallback under
  separation.
- Geometry caching across replicas + complement dedup fix + exact prefix reuse of Kmax-NN lists.
- rddensity on frozen signed distance; honest split as the primary post-selection guarantee
  (Codex: cleaner than cross-fitted-null claims); step-down inference offered but labeled
  approximate (no subset-pivotality proof).
- Overlap-aware aggregation via influence-function/bootstrap covariance (not overlap fraction);
  disjointness (VKNN) as the simple default.
- Buffered predictive **stacking** (refit hyperparameters within folds) instead of softmax-PLP;
  one model label per posterior draw.
- Hierarchical stage-1 noise: Normal likelihood + chi-square measurement model on SE² (not both
  t and latent-variance).
- Exhaustive per-dimension subset enumeration as the exact MDSS baseline at small cardinality;
  corrected single-Δ profile GLR as the fast path; staggered-adoption via group-time ATT
  backend.
- Weak-IV-robust (AR/Fieller) local inference at discovered cutoffs.

Rejected/downgraded:
- Exact conditional hypergeometric test as scan-wide replacement (conditions away the estimated
  propensity offset; fine as a fixed-split diagnostic only).
- Within-neighborhood label permutation (invalid under spatial trends; use null-model bootstrap).
- "Recompute index sets after recentering" (unnecessary — projection leaves the separating
  hyperplane unchanged); keep the two real Alg-1 code fixes (threshold first candidate,
  explicit train_data).
- The audit's blanket ANN refutation was too broad (Codex): ANN acceptable if replicas share the
  same approximation; keep as opt-in for scale.

## 4. Overall reliability assessment

Codex's overall take (§6 of its file) matches ours: the audit's *diagnoses* are largely right
(37 AGREE / ~14 PARTIALLY / 2 DISAGREE of 56), its proposed *repairs* needed refinement in ~10
places, and its two false positives shared a single root cause (text extraction). The
cross-model second opinion added 39 genuinely new issues, of which three are high-impact for
implementation (#1 reverse-neighbor variance, #12 Algorithm 6 incumbent, #28 mixture intervals).
Both documents remain authoritative for details; this file governs where they disagree.
