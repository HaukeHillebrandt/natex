# Phase 2 implementation plan — RDD backtests, synthetic benchmarks, scaling engineering

**Repo:** `/Users/haukehillebrandt/dev/natex` (always quote paths; other referenced paths contain spaces).
**Governing math (wins all conflicts):** `docs/math_audit_final.md` (same file at
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/docs/notes/math_audit_final.md"`).
**Design spec:** `".../RDD/docs/superpowers/specs/2026-07-10-natex-design.md"` §6d, §8, §9 (phase 2), §10.
**Method notes:** `".../RDD/docs/notes/read_kdd2018.md"` (synthetic DGP Eqs 17–22, NIG Eq 16, evaluation
protocol, real-data results tables), `".../RDD/docs/notes/review_data-inventory.md"` (dataset truths,
column mappings, quirks).
**Local data root:** env `NATEX_DATA` (currently
`"/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data"`).

## Phase objective (spec §9 phase 2, task #4)

1. **All remaining RDD backtest rows of spec §8 pass** (test_score_2012 already done in phase 1):
   - AcademicProbation LSO 2010 — fuzzy RDD at `dist_from_cut = 0`; correct forcing variable ranked #1 of 4.
   - ED visits (ADG 2012) — fuzzy RDDs at ages 19 & 23 (`months_23 = −48` and `0`); both among top discoveries.
   - Inpatient visits — age-23 cutoff on 73 cells (small-n robustness).
   - Egger–Köthenbürger 2010 — ≥2 statutory population thresholds discovered (**stretch goal** per spec §10:
     documented failure is acceptable, silent failure is not).
2. **Synthetic benchmarks** reproducing the KDD-2018 ch.5 evaluation curves: NIG vs ζ and power at α=0.05
   across polynomial orders, τ̂ convergence, Bernoulli-vs-Normal comparison on binary T, label-noise
   protocol P(T_ρ=T)=ρ. Small seeded versions run in CI; full curves via a benchmarks/ script.
3. **Scaling engineering** (spec §6d; needed to make the 44k-row LSO backtest tractable): geometry caching
   across randomization replicas, Kmax-NN prefix reuse, vectorized Bernoulli LLR across splits,
   homogeneous-treatment fast path, coarse-to-fine scan that always reports what was and wasn't searched.
4. Supporting hardening: dataset registry/loaders keyed on `NATEX_DATA`, NaN-outcome-tolerant estimators
   (LSO outcomes are 13–56% missing), discovery clustering for multi-cutoff assertions.

## House rules (bind every task)

Python ≥3.11; core deps only numpy/scipy/pandas/scikit-learn/typer/pydantic — everything else optional
extras with tests that skip gracefully; CI (3.11–3.14) must stay green; one `numpy.random.Generator`
through every stochastic call (same seed ⇒ same result); discovery never reads the outcome `y`; NaN never
0.0 on failure; no bare `except`; never commit datasets; conventional commit after every green cycle.
`uv run pytest -q` excludes backtests; run backtests with `-m backtest` and `NATEX_DATA` set.

**TDD discipline for every task:** write the failing test(s) first, run them to confirm failure, implement,
run the full suite (`uv run pytest -q` and `uv run ruff check src tests`), then commit.

## Current interfaces built upon (do not break)

- `natex.data.spec.Dataset` / `DatasetSpec` — `Dataset(df, spec)`, `Dataset.from_csv(path, treatment,
  outcome=None, forcing=None, covariates="auto", time=None)`; properties `n, T, y, Z, Z_std, X,
  treatment_is_binary`. Drops NaN rows in scan columns only (never `y`).
- `natex.rdd.lord3.lord3_scan(dataset, k=50, model="auto", degree=1, rng=None) -> LoRD3Result`;
  `Discovery(center_index, k, llr, normal, members, group1, p_value, extras)`;
  `fit_treatment_model(X, T, model, degree)` (ridge-logistic separation guard for binary T).
- `natex.scan.neighborhoods.knn_indices(z_std, k)`, `candidate_partitions(cz) -> (G, keep)`,
  `local_residual_variance(r, idx)`.
- `natex.scan.statistics.normal_llr_all_splits(r, w, G)`, `bernoulli_llr_all_splits(t, eta, G)`,
  `fit_log_odds_offset(t, eta)`, `offset_log_lik(theta, t, eta)`.
- `natex.validate.randomization.randomization_test(dataset, scan_result, Q=99, rng, scan_kwargs) ->
  RandomizationReport(p_value, observed_max_llr, null_max_llrs, q)` (+1-rank p, fitted-null bootstrap).
- `natex.validate.placebo.placebo_tests`, `hc1_ols(Xmat, yvec)`, `signed_distance(dataset, d)`;
  `natex.validate.density.density_test`; `natex.validate.honest.honest_split`.
- `natex.estimate.local2sls.local_2sls(dataset, d) -> EffectEstimate(tau, se, ci, method,
  first_stage_jump, first_stage_t, weak_instrument)`; `wald_estimate(dataset, d)`.
- `natex.rdd.metrics.normalized_information_gain(true_D, members, group1)`.
- `natex.data.synthetic.make_synthetic(n, px=2, pz=2, zeta=3.0, tau=2.0, kind="binary",
  discont="square", rng) -> (Dataset, D)`.
- CLI `natex discover` (typer app in `natex/cli.py`).

All signature changes below are **additive** (new keyword args with defaults, new dataclass fields with
defaults) so phase-1 tests keep passing unmodified.

---

## Task 1 — Commit this plan; dataset registry and loaders

**First action of this task: `git add docs/plans/phase-2.md && git commit -m "docs: phase 2 implementation
plan"` — the plan file is committed before any code.**

**Create:** `src/natex/data/registry.py`, `tests/test_registry.py`, `tests/backtests/conftest.py`.
**Modify:** `src/natex/data/__init__.py` (re-export), `src/natex/__init__.py` (export `load_dataset`).

Interface (exact):

```python
# src/natex/data/registry.py
from dataclasses import dataclass, field
from pathlib import Path
from natex.data.spec import Dataset

@dataclass(frozen=True)
class DatasetInfo:
    name: str
    relpath: str                    # main CSV relative to the data root
    glob_fallback: str | None       # e.g. "EggerKoethenbuerger_AEJ_Data*.csv" (handles " (1)" suffix)
    treatment: str
    outcome: str | None             # default outcome; loaders accept an override
    forcing: tuple[str, ...]
    covariates: tuple[str, ...]     # explicit, never "auto" (reproducibility)
    n_rows: int | None              # expected data rows; None = don't check
    source: str                     # human fetch instructions (URL + landing page)
    notes: str = ""

REGISTRY: dict[str, DatasetInfo]   # keys exactly:
# "test_score_2012", "academic_probation", "ed_visits", "inpatient_visits", "egger_koethenbuerger"

@dataclass(frozen=True)
class DatasetStatus:
    name: str
    found: bool
    path: Path | None
    n_rows: int | None
    ok: bool
    message: str

def data_root(root: str | Path | None = None) -> Path
    # root arg > env NATEX_DATA; raises RuntimeError naming NATEX_DATA and the expected layout if unset.
def locate(name: str, root: str | Path | None = None) -> Path
    # resolves relpath, falls back to glob_fallback; FileNotFoundError message includes REGISTRY[name].source.
def verify(name: str, root: str | Path | None = None) -> DatasetStatus
def load_dataset(name: str, root: str | Path | None = None, outcome: str | None = "default") -> Dataset
    # outcome="default" uses DatasetInfo.outcome; None loads without outcome; any str overrides.
```

Registry entries (from `data_inst.json` files and `review_data-inventory.md`; never committed data):

| name | relpath | treatment | outcome | forcing | covariates | n_rows |
|---|---|---|---|---|---|---|
| test_score_2012 | `test_score_2012/RDD_Guide_Dataset_0.csv` | `treat` | `posttest` | `age, pretest` | `gender, sped, frlunch, esol, black, white, hispanic, asian, age, pretest` | 2766 |
| academic_probation | `AcademicProbation_LSO_2010/data_orig.csv` | `probation_year1` | `GPA_year2` | `dist_from_cut, hsgrade_pct, totcredits_year1, age_at_entry` | forcing + `sex, bpl_north_america, mtongue, loc_campus1, loc_campus2` | 44362 |
| ed_visits | `ED_visits/P03_ED_Analysis_File.csv` | `priv_all` | `all` | `months_23` | `months_23` | 161 |
| inpatient_visits | `Inpatient_visits/P10_Inpatient_CSV_File.csv` | `TOT_priv_ALL` | `TOT_ALL` | `months_23` | `months_23` | 73 |
| egger_koethenbuerger | `EggerKoethenbuerger_AEJ_Data.csv` (glob fallback `EggerKoethenbuerger_AEJ_Data*.csv`) | `rcsize` | `exptot` | `log_pop` (derived) | `log_pop` | 43175 |

Loader specifics: `egger_koethenbuerger` drops rows with missing/nonpositive `wpop`, adds derived column
`log_pop = np.log(wpop)` before building the `Dataset` (population is heavily right-skewed; statutory
thresholds are multiplicative). `academic_probation` passes the raw frame through — `Dataset` already
one-hot-encodes string covariates (`sex`, `mtongue`) and keeps NaN outcomes (scan-column NaN dropping never
touches `y`). Sources: MDRC RDD practice dataset (Jacob et al. 2012, mdrc.org); LSO 2010 and
Anderson/Dobkin/Gross 2012 from the AEJ:Applied data archives (openICPSR); Egger–Köthenbürger 2010 AEJ:Applied
archive. No auto-download in this phase (archive URLs are login-gated); `source` strings carry instructions.

`tests/backtests/conftest.py`: helper `load_or_skip(name: str, **kwargs) -> Dataset` that calls
`pytest.skip` with the registry `source` message when `NATEX_DATA` is unset or the file is missing.
Migrate `tests/backtests/test_test_score.py` to use `load_dataset("test_score_2012")` (assert its spec
matches the phase-1 hand-built one — same treatment/outcome/forcing/covariates).

**Tests first (`tests/test_registry.py`, all against a `tmp_path` fake root — no real data in CI):**
- `test_registry_names`: `set(REGISTRY) == {the 5 names}` and every entry has nonempty `source`.
- `test_data_root_requires_env`: with `monkeypatch.delenv("NATEX_DATA", raising=False)`,
  `data_root(None)` raises `RuntimeError` whose message contains `"NATEX_DATA"`.
- `test_load_dataset_maps_columns`: write a 30-row fake `test_score_2012` CSV with the real header into
  `tmp_path`; `load_dataset("test_score_2012", root=tmp_path)` returns a `Dataset` with
  `spec.treatment == "treat"`, `spec.forcing == ["age", "pretest"]`, `ds.n == 30`.
- `test_locate_glob_fallback`: create `tmp_path/"EggerKoethenbuerger_AEJ_Data (1).csv"` (19 real columns,
  5 rows, positive `wpop`); `locate` finds it; `load_dataset` adds `log_pop` and
  `spec.forcing == ["log_pop"]`.
- `test_verify_missing_and_row_mismatch`: absent file → `found=False, ok=False`, message contains the
  source string; present file with wrong row count → `found=True, ok=False`, message mentions expected count.
- `test_outcome_override_and_none`: `load_dataset(..., outcome=None)` gives `ds.y is None`;
  `outcome="left_school"`-style override changes `spec.outcome` (use fake academic_probation CSV).

**Commit:** `feat: dataset registry and loaders for the five RDD benchmark datasets`

---

## Task 2 — Estimators tolerate missing outcomes (never 0.0, never crash)

LSO outcomes are 13–56% NaN; `local_2sls`/`wald_estimate` currently propagate NaN through the algebra.
Audit rule: failed computations return NaN, never zero; no silent wrong numbers.

**Create:** `tests/test_estimate_missing_outcome.py`.
**Modify:** `src/natex/estimate/local2sls.py`.

Interface changes (additive):

```python
@dataclass
class EffectEstimate:
    ...existing fields...
    n_used: int = 0        # members with finite y actually used
```

Both estimators: mask `m` to members with `np.isfinite(ym)`; recompute `s`, `g`, controls on the masked
set. If `n_used < 8`, or either side of the (re-masked, re-oriented) split is empty, or `dt == 0` in Wald:
return an `EffectEstimate` with `tau=se=nan`, `ci=(nan, nan)`, `first_stage_jump/t` still computed from the
finite-`T` rows where possible (else NaN), `weak_instrument=True`, `n_used` set. No exceptions for NaN `y`;
`ValueError` stays for `dataset.y is None`.

**Tests first:**
- `test_partial_nan_outcome_close_to_clean`: synthetic `make_synthetic(n=1500, zeta=4.0, kind="real",
  rng=default_rng(0))`, scan `k=40`, take top discovery; copy dataset, set `y` to NaN for a seeded random
  30% of `members`; both estimates satisfy `abs(tau_masked - tau_clean) < 1.0` and
  `est.n_used == (# finite-y members)`.
- `test_all_nan_outcome_returns_nan_not_zero`: all `y[members] = nan` → `math.isnan(est.tau)`,
  `math.isnan(est.se)`, both CI ends NaN, `est.weak_instrument is True`, `est.n_used == 0`, and the call
  raises nothing. Explicitly assert `est.tau != 0.0` (NaN comparison is False — also assert via `isnan`).
- `test_one_sided_after_masking_returns_nan`: poison `y` so only group-1 members remain finite → NaN tau.
- Existing `tests/test_estimate.py` must pass unchanged (defaults preserve behavior on finite data;
  `n_used == members.size` there).

**Commit:** `fix: estimators drop non-finite outcomes and return NaN (never 0.0) when underdetermined`

---

## Task 3 — Vectorized Bernoulli LLR kernel + homogeneous fast path

Current `bernoulli_llr_all_splits` runs a Python-level bracketed Newton per split side — O(n·k) Newton
loops makes the 44k-row LSO scan take hours. Vectorize Newton across all split-sides of a neighborhood at
once. Audit items preserved: boundary likelihood suprema for pure groups (item 21), bracketed Newton in
log-odds (item 22), LLR ≥ 0, degenerate splits exactly 0.0.

**Create:** `tests/test_statistics_bernoulli_vectorized.py`.
**Modify:** `src/natex/scan/statistics.py`, `src/natex/rdd/lord3.py` (homogeneous fast path).

Interface (exact):

```python
# statistics.py — same public name, new implementation:
def bernoulli_llr_all_splits(t: np.ndarray, eta: np.ndarray, G: np.ndarray) -> np.ndarray
def bernoulli_llr_all_splits_reference(t, eta, G) -> np.ndarray   # the phase-1 loop, kept for parity tests
def fit_log_odds_offsets(t: np.ndarray, eta: np.ndarray, M: np.ndarray) -> np.ndarray
    # M: bool (k, m) membership mask, one column per group; returns theta (m,), +/-inf for pure columns.
def masked_offset_log_lik(theta: np.ndarray, t, eta, M) -> np.ndarray
    # sum_i M_ij * (t_i * z_ij - log1pexp(z_ij)), z_ij = eta_i + theta_j; 0.0 at +/-inf boundary
    # for the matching pure column (all-ones / all-zeros), -inf otherwise.
```

Vectorized Newton: per-column state `theta (m,)`, brackets `lo/hi` initialized to (−30, 30); per iteration
compute `P = expit(eta[:, None] + theta[None, :])` masked by `M`, `score_j = Σ_i M_ij (t_i − P_ij)`,
`info_j = Σ_i M_ij P_ij (1 − P_ij)`; Newton step clipped to the bracket with bisection fallback (same logic
as the scalar version, per-column); converged when `|score| < 1e-11`; max 200 iterations. Pure columns are
excluded from iteration and assigned ±inf up front. `bernoulli_llr_all_splits` calls
`fit_log_odds_offsets` twice (masks `G` and `~G`), computes `ll1` via `masked_offset_log_lik`, subtracts a
single scalar `ll0`, clamps at 0, and forces degenerate (one-sided) columns to exactly 0.0.

Homogeneous fast path (in `lord3_scan`): if `T[members]` is constant, every split has both sides pure, so
`ll1 = 0` for all splits and `llr = −ll0 = 0` for all splits (the null's boundary supremum is also 0).
Skip the kernel, record no discovery for that center (identical output to the slow path, which scores all
splits 0.0 — assert this equivalence in tests). This also keeps null replicas consistent.

**Tests first:**
- `test_vectorized_matches_reference`: seeded `default_rng(0)`; k=24 points, eta ~ N(0,1), t ~ Bernoulli,
  G = 40 random columns **plus** hand-built pure-group columns (one side all-1, one side all-0) and
  near-degenerate columns (single point on a side): `np.allclose(new, reference, atol=1e-8)`.
- `test_llr_nonnegative_and_degenerate_zero`: all outputs ≥ 0; all-True/all-False columns exactly 0.0.
- `test_pure_group_boundary_supremum_finite`: sharp split (t == g) scores finite and strictly greater
  than every mixed split of the same neighborhood (boundary suprema, audit item 21).
- `test_homogeneous_neighborhood_scores_zero`: t all-ones, mixed eta → every split exactly 0.0 via both
  the reference and the fast path; `lord3_scan` on a dataset with a constant-T region produces identical
  `discoveries` list with fast path on (it is always on — compare against monkeypatched slow scoring).
- `test_scan_end_to_end_parity`: `lord3_scan` on `make_synthetic(n=600, zeta=3.0, kind="binary",
  rng=default_rng(2))`, k=30 — top-5 `(center_index, llr)` identical (llr to 1e-8) before/after by
  comparing against `bernoulli_llr_all_splits_reference` monkeypatched in.
- Perf guard (loose, not flaky): scoring 200 random neighborhoods (k=50, m=49) completes in < 5 s.

**Commit:** `perf: vectorize Bernoulli LLR Newton across splits with boundary suprema preserved`

---

## Task 4 — Scan geometry cache: Kmax-NN prefix reuse, replica reuse, center subsets

Audit §3 (adopted improvements): "Geometry caching across replicas + complement dedup fix + exact prefix
reuse of Kmax-NN lists". Geometry (kNN + deduped partitions) depends only on `Z_std`, which is identical
across all Q null replicas — computing it Q+1 times is the dominant avoidable cost of
`randomization_test`.

**Create:** `src/natex/scan/geometry.py`, `tests/test_geometry.py`.
**Modify:** `src/natex/rdd/lord3.py` (`lord3_scan(..., geometry=None, centers=None)`),
`src/natex/validate/randomization.py` (build once, reuse), `src/natex/scan/__init__.py`.

Interface (exact):

```python
# src/natex/scan/geometry.py
@dataclass
class ScanGeometry:
    k: int
    idx: np.ndarray                                  # (n, k) own-neighborhood indices, self first
    _partitions: dict[int, tuple[np.ndarray, np.ndarray]]  # center -> (G, keep), filled lazily

    def partitions_for(self, i: int, Z_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]
        # lazy candidate_partitions((Z_std[idx[i]] - Z_std[i])), cached. Memory note: ~n*k^2 bits
        # when fully populated (44k x 50 x 49 booleans ~ 108 MB) — acceptable; document in docstring.
    def shrink(self, k2: int) -> "ScanGeometry"
        # exact Kmax-NN prefix: ScanGeometry(k2, idx[:, :k2], fresh empty cache); requires k2 <= k.

def build_geometry(Z_std: np.ndarray, k: int) -> ScanGeometry   # knn_indices + empty cache
```

`lord3_scan` additions: `geometry: ScanGeometry | None = None` (built internally when None) and
`centers: np.ndarray | None = None` (scan only these center indices; `Discovery.center_index` stays a
global dataset index). `LoRD3Result` gains `centers: np.ndarray | None = None` (dataclass field with
default). Normal-model `sigma2` still uses the full `geometry.idx` (every member's own-kNN variance is
needed even when scanning a center subset).

`randomization_test` additions: `geometry: ScanGeometry | None = None`, `centers: np.ndarray | None = None`;
builds geometry once and passes it into the observed-data re-scan path and every replica scan. Replica
draw order is unchanged, so p-values are bit-identical with/without caching.

**Tests first (`tests/test_geometry.py`):**
- `test_idx_matches_knn_indices`: `build_geometry(Z, k).idx` array-equal to `knn_indices(Z, k)` on
  `make_synthetic(n=400, rng=default_rng(0))` (continuous z, no ties).
- `test_shrink_prefix_exact`: `build_geometry(Z, 40).shrink(15).idx` array-equal `knn_indices(Z, 15)`.
- `test_scan_with_geometry_identical`: `lord3_scan(ds, k=25)` vs `lord3_scan(ds, k=25,
  geometry=build_geometry(ds.Z_std, 25))` — identical discovery count and top-10
  `(center_index, llr, group1)`.
- `test_centers_subset`: `centers=np.arange(0, n, 7)` → every returned `center_index` is in the subset,
  and each equals the corresponding full-scan discovery for that center (same llr).
- `test_randomization_bitwise_parity_and_single_knn_build`: monkeypatch
  `natex.scan.neighborhoods.knn_indices` with a counting wrapper; `randomization_test(ds, res, Q=5,
  rng=default_rng(3))` calls it exactly once, and its `p_value` and `null_max_llrs` are identical to the
  phase-1 path re-run with the same seed (regression-pin the phase-1 values inside the test by computing
  both paths: cached vs a `geometry=None`-per-replica fallback kept private for the test via explicit
  per-replica `build_geometry` calls).

**Commit:** `perf: scan geometry cache with Kmax prefix reuse; randomization replicas share geometry`

---

## Task 5 — Coarse-to-fine scan

Spec §6d (subsample → localize → rescan at full resolution near candidates) and §6b (always report what
was and wasn't searched). Deterministic given the seed. This is what makes LSO (44,362 rows) tractable.

**Create:** `src/natex/scan/coarse.py`, `tests/test_coarse.py`.
**Modify:** `src/natex/scan/__init__.py`, `src/natex/__init__.py` (export `coarse_to_fine_scan`).

Interface (exact):

```python
# src/natex/scan/coarse.py
@dataclass
class CoarseToFineResult:
    result: LoRD3Result            # fine-stage discoveries (full-resolution, subset of centers)
    coarse_result: LoRD3Result
    fine_centers: np.ndarray       # dataset indices scanned at full resolution
    frac_centers_scanned: float    # len(unique(coarse ∪ fine centers)) / n
    params: dict                   # n_coarse, top_m, radius_mult, k, model, degree, seed-note

def coarse_to_fine_scan(
    dataset: Dataset,
    k: int = 50,
    n_coarse: int = 2000,
    top_m: int = 20,
    radius_mult: float = 2.0,
    model: str = "auto",
    degree: int = 1,
    rng: np.random.Generator | None = None,   # required, ValueError if None (house rule)
    geometry: ScanGeometry | None = None,
) -> CoarseToFineResult
```

Algorithm: (1) build (or accept) full geometry once — the kNN query is cheap even at 44k; the savings are
in per-center partition/LLR work; (2) coarse stage: `centers = rng.choice(n, size=min(n_coarse, n),
replace=False)`, `lord3_scan(..., centers=centers, geometry=geometry)`; (3) localization: for each of the
`top_m` coarse discoveries, fine-center set = all points whose `Z_std` lies within
`radius_mult × r_k(center)` of the discovery center, where `r_k` is the distance to its k-th neighbor
(query the same KD-tree); (4) fine stage: `lord3_scan(..., centers=union_of_fine_sets)`; (5) return with
`frac_centers_scanned` and full `params` — the never-silently-truncate contract: callers (CLI, results
bundle) must be able to state coverage.

**Tests first (`tests/test_coarse.py`):**
- `test_finds_planted_boundary_cheaply`: `make_synthetic(n=6000, zeta=4.0, kind="real",
  rng=default_rng(0))`, `coarse_to_fine_scan(ds, k=40, n_coarse=600, top_m=10, rng=default_rng(1))`:
  top fine discovery has `normalized_information_gain > 0.4` against true `D`, and
  `frac_centers_scanned < 0.3`.
- `test_deterministic`: two runs with `default_rng(1)` → identical top `(center_index, llr)` and identical
  `fine_centers`.
- `test_reports_coverage`: `0 < frac_centers_scanned <= 1`, `params` contains keys
  `{"n_coarse", "top_m", "radius_mult", "k", "model", "degree"}`, `fine_centers` is sorted unique.
- `test_rng_required`: `rng=None` raises `ValueError`.
- `test_small_n_degenerates_to_full`: n=300 < n_coarse → every point is a coarse center and result matches
  plain `lord3_scan` top-5 exactly.

**Commit:** `feat: seeded coarse-to-fine scan with explicit search-coverage reporting`

---

## Task 6 — Synthetic DGP fidelity (KDD Eqs 17–22) + label-noise protocol

Faithful options for the paper's generator, off by default so phase-1 tests are untouched. Audit context:
the binary-treatment log-odds shift correction is already in (`(ζ/2)(2D−1)` additive in log-odds — the
printed `+μ` with `μ = exp(±ζ/2)` is the typo the audit confirmed); this task adds Eq 17's uniform
confounder, Eq 18's random per-dimension boundaries, Eq 19's covariate-driven heteroskedastic noise, and
the KDD noise-injection protocol P(T_ρ = T) = ρ.

**Create:** `tests/test_synthetic_fidelity.py`.
**Modify:** `src/natex/data/synthetic.py`.

Interface (exact, all-new kwargs default to phase-1 behavior):

```python
def make_synthetic(
    n: int, px: int = 2, pz: int = 2, zeta: float = 3.0, tau: float = 2.0,
    kind: str = "binary", discont: str = "square",
    rng: np.random.Generator | None = None,
    boundary: float | str = 0.5,          # scalar, or "random": b_j ~ U(0,1) resampled (seeded) until
                                          # region mass in [min_region_frac, 1 - min_region_frac]
    min_region_frac: float = 0.05,        # documented deviation: guards degenerate empty corners
    heteroskedastic: bool = False,        # Eq 19: eps_T, eps_p, eps_y ~ N(0, mean_j(x_ij)) per point
    confounder: str = "normal",           # "normal" (phase 1: N(0,0.5)) | "uniform" (Eq 17: U(0,1))
) -> tuple[Dataset, np.ndarray]

def draw_confounder(n: int, kind: str, rng: np.random.Generator) -> np.ndarray  # exported for tests

def inject_label_noise(T: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray
    # requires binary T; flips each label with prob (1 - rho)  =>  P(T_rho = T) = rho exactly;
    # ValueError for rho outside [0.5, 1] or non-binary T; never mutates the input.
```

**Tests first:**
- `test_defaults_unchanged`: same seed → `make_synthetic(n=500, rng=default_rng(0))` produces a DataFrame
  equal (`pd.testing.assert_frame_equal`) to the value produced with all new kwargs at their defaults
  passed explicitly (guards accidental rng-consumption reordering).
- `test_random_boundary_mass_bounds`: `boundary="random"`, 20 seeds, n=4000: every region mass in
  `[0.05, 0.95]` and boundaries differ across seeds.
- `test_heteroskedastic_variance_tracks_x`: kind="real", ζ=0, τ=0, n=20000, heteroskedastic=True: OLS of
  T on x, then Spearman correlation between squared residuals and `mean_j(x_ij)` > 0.1; with
  heteroskedastic=False the same statistic < 0.05.
- `test_uniform_confounder_range`: `draw_confounder(10000, "uniform", rng)` in [0, 1] with mean in
  (0.45, 0.55); `"normal"` has negative values; unknown kind raises `ValueError`.
- `test_inject_label_noise_exact_rate`: n=20000 binary T: rho=1 → identical array (and not the same
  object); rho=0.8 → agreement fraction in (0.78, 0.82); rho=0.5 → in (0.48, 0.52); determinism under the
  same seed; `rho=0.4` and continuous T raise `ValueError`.

**Commit:** `feat: KDD-faithful synthetic options (Eq 17-19) and label-noise injection protocol`

---

## Task 7 — Benchmark harness (NIG/power/τ̂ curves) + CI-small assertions

Reproduce the KDD ch.5 evaluation curves (spec §8 synthetic list for this phase: NIG vs ζ across
polynomial orders; plus power at α=0.05, τ̂ convergence, Bernoulli-vs-Normal, label noise). Full curves are
a script; small seeded slices run in CI. The changepoint-baseline comparison (KDD Fig 9) is **out of
scope** for phase 2 (not in the spec's benchmark list). p-values are +1-rank fitted-null Monte Carlo
(audit item 1 language: never "exact").

**Create:** `src/natex/benchmarks.py`, `benchmarks/run_nig_curve.py`, `benchmarks/README.md`,
`tests/test_benchmarks_small.py`.

Interface (exact):

```python
# src/natex/benchmarks.py
def nig_power_curve(
    kind: str,                              # "real" | "binary"
    zetas: Sequence[float],
    n_experiments: int = 50,
    n: int = 1000, k: int = 50,
    degrees: Sequence[int] = (1, 2, 4),
    models: Sequence[str] = ("auto",),      # ("normal","bernoulli") for the Fig-7 comparison
    Q: int = 99, alpha: float = 0.05, tau: float = 5.0,
    boundary: float | str = "random", heteroskedastic: bool = True, confounder: str = "uniform",
    seed: int = 0,
) -> pd.DataFrame
    # one row per (zeta, degree, model): columns
    # [zeta, degree, model, kind, nig_mean, nig_se, power, p_mean, tau_2sls_median, tau_wald_median,
    #  n_experiments] — power = fraction of experiments with +1-rank p <= alpha; NaN-safe medians
    # (np.nanmedian); every experiment uses child generators spawned from default_rng(seed) so rows are
    # independent of evaluation order.

def label_noise_curve(
    rhos: Sequence[float], n_experiments: int = 25, n: int = 2000, k: int = 50,
    zeta_sharp: float = 8.0, seed: int = 0,
) -> pd.DataFrame                            # columns [rho, nig_mean, nig_se, n_experiments]
    # sharp-ish binary synthetic (T = D via large zeta), inject_label_noise per experiment, top-1 NIG.
```

`benchmarks/run_nig_curve.py`: stdlib-argparse script; writes
`benchmarks/out/nig_curve_{kind}.csv` (and `label_noise.csv`); if matplotlib is importable, also writes
PNG line charts (NIG vs ζ per degree; power vs ζ) — wrapped in `importlib.util.find_spec("matplotlib")`
check, silently skipping plots otherwise (matplotlib stays an optional extra). `benchmarks/out/` is
gitignored. `benchmarks/README.md` states the paper protocol (50 experiments/ζ, n=1000, k=50, τ=5, ζ up to
2.5 real / ~5 binary) and the expected qualitative shapes (NIG and power increase with ζ; order-2/4 mildly
worse; Bernoulli ≥ Normal on binary T; τ̂ overestimates at low ζ — from `read_kdd2018.md`).

**Tests first (`tests/test_benchmarks_small.py` — runs in default CI, keep total < ~90 s, all seeded):**
- `test_curve_schema_and_monotonicity`: `nig_power_curve("real", zetas=(0.0, 4.0), n_experiments=3, n=600,
  k=40, degrees=(1,), Q=19, seed=0)` → exactly 2 rows; required columns present;
  `nig_mean(ζ=4) ≥ nig_mean(ζ=0) + 0.2`; `power(ζ=4) ≥ 2/3`; `power(ζ=0) ≤ 1/3`; `p_mean(ζ=0) > 0.25`.
- `test_tau_recovery_strong_signal`: from the ζ=4 row of a `nig_power_curve("real", zetas=(4.0,),
  n_experiments=3, n=1200, k=50, tau=5.0, Q=19, seed=1)` run: `abs(tau_2sls_median − 5) < 1.5`.
- `test_bernoulli_dominates_normal_on_binary`: `nig_power_curve("binary", zetas=(4.0,), n_experiments=3,
  n=900, k=40, models=("normal", "bernoulli"), Q=19, seed=2)`:
  `nig_mean(bernoulli) ≥ nig_mean(normal) − 0.05` (qualitative Fig-7 direction with slack).
- `test_label_noise_monotone`: `label_noise_curve(rhos=(0.6, 1.0), n_experiments=3, n=800, k=40, seed=3)`:
  `nig_mean(ρ=1.0) ≥ nig_mean(ρ=0.6) + 0.1` and `nig_mean(ρ=1.0) > 0.5`.
- `test_determinism`: same call twice → `pd.testing.assert_frame_equal`.
- Thresholds above were chosen with wide margins; seeds are fixed, so the tests are deterministic — if a
  threshold fails during implementation, first suspect the implementation, only then widen with a comment.

**Commit:** `feat: synthetic benchmark harness (NIG/power/tau curves, label noise) with CI-small tests`

---

## Task 8 — Discovery clustering (multi-cutoff support)

The ED (two cutoffs) and Egger (≥2 thresholds) backtests need "distinct discovered locations", but
overlapping neighborhoods produce near-duplicate discoveries (KDD's own top-10 averaging is heuristic).
Add a deterministic greedy clusterer.

**Create:** `tests/test_metrics_clusters.py`.
**Modify:** `src/natex/rdd/metrics.py`.

Interface (exact):

```python
@dataclass
class DiscoveryCluster:
    representative: Discovery      # highest-LLR member
    center_z: np.ndarray           # raw-z of representative center (copy)
    size: int
    max_llr: float

def cluster_discoveries(
    result: LoRD3Result,
    Z_raw: np.ndarray,             # dataset.Z (raw units, so tolerances are interpretable)
    tol: float | np.ndarray,       # per-dimension absolute tolerance (broadcast if scalar)
    top: int | None = None,        # cluster only the first `top` discoveries (default: all)
) -> list[DiscoveryCluster]
    # Greedy in descending LLR (result.discoveries is already sorted): assign a discovery to the first
    # existing cluster whose representative center is within `tol` on EVERY z-dimension (Chebyshev after
    # per-dim scaling), else open a new cluster. Returned list ordered by max_llr descending.
```

**Tests first:**
- `test_merges_within_tol_and_orders`: hand-built `LoRD3Result` with discoveries at 1-D raw centers
  `[0.0, 0.5, 10.0, 10.4, 30.0]` and LLRs `[9, 8, 7, 6, 5]`, `tol=1.0` → 3 clusters, sizes `[2, 2, 1]`,
  representatives at `0.0, 10.0, 30.0`, `max_llr == [9, 7, 5]`.
- `test_per_dimension_tol`: 2-D centers where dim-0 is within tol but dim-1 is not → separate clusters
  with `tol=np.array([5.0, 0.1])`.
- `test_top_limits_input`: `top=2` clusters only the two best discoveries.
- `test_empty_result`: no discoveries → `[]`.

**Commit:** `feat: greedy discovery clustering for multi-cutoff assertions`

---

## Task 9 — Backtests: ED visits (cutoffs at ages 19 and 23) and inpatient visits

Spec §8 rows 3–4. Both files are aggregated age-in-month cells with continuous treatment shares (fuzzy
RDD, normal model). Paper protocol: cubic background f, strongest RDD at the 23rd birthday, age-19
recovered (`months_23 = −48`); Table 4 detected locations {16.83, 19, 23.25} ⇒ months {−74, −48, +3}.
Audit framing: density test on a uniform month grid is a falsification-only check (item 6) and the
covariate set equals the forcing set here, so the placebo battery is trivially empty — assert and document
both facts rather than skipping them silently.

**Create:** `tests/backtests/test_ed_inpatient.py`.

Test spec (all `pytestmark = pytest.mark.backtest`, data via `load_or_skip`):

- `ed` fixture: `load_or_skip("ed_visits")` (n=161, T=`priv_all`, forcing `months_23`).
- `test_ed_recovers_both_cutoffs`: `lord3_scan(ds, k=25, degree=3, rng=default_rng(0))` (model
  auto-selects "normal"); `clusters = cluster_discoveries(res, ds.Z, tol=6.0, top=30)`; assert a cluster
  with `|center − 0| ≤ 6` months exists among the top 3 clusters **and** a cluster with
  `|center + 48| ≤ 6` exists among the top 5 clusters.
- `test_ed_scan_significant`: `randomization_test(ds, res, Q=99, rng=default_rng(1),
  scan_kwargs={"k": 25, "degree": 3})` → `p_value ≤ 0.05`.
- `test_ed_density_uniform_grid_passes`: `density_test` on the top discovery → `p_value > 0.05`
  (months grid is uniform by construction — negative control for the falsification test).
- `test_ed_effect_direction`: `local_2sls` on the best near-zero-cluster representative with outcome
  `all`: first stage not weak (`weak_instrument is False`) and `tau > 0` (insurance loss at 23 reduces ED
  visits and the private-insurance share drops — both jump down ⇒ positive τ of visits on `priv_all`;
  the paper's ~1.6% visit drop / ~1.5 pp coverage drop implies τ ≈ 0.01·all/0.015·1 — assert only the
  sign plus `est.n_used == 25` finite cells; magnitudes on 161 aggregate cells are not the paper's
  individual-level estimand and the docstring must say so).
- `inpatient` fixture: `load_or_skip("inpatient_visits")` (n=73).
- `test_inpatient_recovers_age23_small_n`: `lord3_scan(ds, k=15, degree=3, rng=default_rng(2))`;
  top cluster (`tol=6.0`) center within ±6 months of 0; `randomization_test(Q=99, rng=default_rng(3))`
  `p ≤ 0.10` (73 cells — power-limited; threshold documented in the test).

Runtime target: both files together < 2 min (n ≤ 161, trivially fast).

**Commit:** `test: ED and inpatient backtests recover age-19/23 insurance cutoffs`

---

## Task 10 — Backtest: Academic Probation (LSO 2010), 44k rows via coarse-to-fine

Spec §8 row 2: fuzzy RDD at `dist_from_cut = 0`; the correct forcing variable must rank #1 of 4. This is
the scale gate for tasks 3–5. Audit item 20 applies: legacy outputs are NOT parity targets; we validate
against the dataset's known truth and the paper's qualitative Table-2/3 results (Bernoulli influence on
GPA-distance 1.0; Lindo GPA_year2 effect ≈ +0.23, LoRD3 2SLS ≈ 0.26).

**Create:** `tests/backtests/test_academic_probation.py`.

Test spec:

- Fixture `ds`: `load_or_skip("academic_probation")` (T=`probation_year1` binary ⇒ Bernoulli model;
  forcing = 4 real columns; string covariates one-hot; NaN outcomes retained).
- `test_recovers_probation_cutoff_and_forcing_rank`:
  `ctf = coarse_to_fine_scan(ds, k=75, n_coarse=3000, top_m=20, rng=default_rng(0))`;
  among `ctf.result.top(10)`: at least one center with raw `|dist_from_cut| < 0.1`; for the top-10
  discoveries, mean absolute normal component of `dist_from_cut` is the largest of the 4 forcing
  dimensions (paper: 1.0 ± 0.0 for Bernoulli). Also assert `ctf.frac_centers_scanned < 0.5` (the scaling
  machinery actually engaged) and that `ctf.params` records the budget.
- `test_scan_significant_on_subsample`: seeded 8000-row subsample (`rng.choice` on the dataframe,
  new `Dataset` with the same spec); full `lord3_scan(k=75)` + `randomization_test(Q=19,
  rng=default_rng(1))` → `p_value ≤ 0.05`. (Randomization on the full 44k×Q is out of compute budget;
  the subsample statement is conditionally valid and documented in the test docstring.)
- `test_effect_sign_matches_lindo`: best near-cutoff discovery from the fine stage; `local_2sls` with
  default outcome `GPA_year2` (13% NaN — exercises task 2): `est.n_used ≥ 40`,
  `0.0 < est.tau < 1.0` (Lindo 0.233; LoRD3-paper 0.255 ± —; generous bracket, sign is the claim),
  `est.weak_instrument is False`.
- Runtime budget: the whole file < ~10 min on an M-series laptop with tasks 3–5 in place. If the fine
  stage exceeds this, reduce `n_coarse` to 2000 before touching `k` (paper used k=100; k=75 keeps the
  partition count manageable — record the final choice in the test docstring).

**Commit:** `test: academic-probation backtest (44k rows) via coarse-to-fine; forcing variable ranked #1`

---

## Task 11 — Backtest: Egger–Köthenbürger multi-threshold (stretch)

Spec §8 row 5 and §10: LoRD3 was not designed for many parallel cutoffs; ≥2 statutory thresholds is the
target; a documented failure is acceptable, silence is not. Bavarian statutory council-size thresholds at
population 1,001 / 2,001 / 3,001 / 5,001 / 10,001 (and above); T = `rcsize` (integer-valued council size,
scanned with the normal model), forcing = `log_pop`.

**Create:** `tests/backtests/test_egger_koethenbuerger.py`.

Test spec:

- Fixture: `load_or_skip("egger_koethenbuerger")`; then restrict to `wpop < 20000` rows (keeps the
  low thresholds from being drowned by the population tail; recompute via a fresh `Dataset` on the
  filtered frame, same spec) — the restriction is part of the test protocol and documented inline.
- `test_multi_threshold_discovery`:
  `ctf = coarse_to_fine_scan(ds, k=100, n_coarse=3000, top_m=30, rng=default_rng(0))`;
  `clusters = cluster_discoveries(ctf.result, np.exp(ds.Z), tol=<15% multiplicative — pass tol on the
  raw-pop scale as 0.15 * threshold via per-cluster check>, top=50)`. Concretely: for thresholds
  `[1001, 2001, 3001, 5001, 10001]`, count how many have at least one of the top-15 cluster
  representatives with `|pop − threshold| / threshold < 0.15`. Assert `count ≥ 2`.
  Decorate with `@pytest.mark.xfail(strict=False, reason="stretch goal, spec §10: LoRD3 not designed for
  many parallel cutoffs")` **only if** it does not pass after reasonable tuning (k ∈ {50, 100},
  pop cap ∈ {10k, 20k}); if it passes, leave it strict and delete the decorator. Either way the outcome
  (pass, or which thresholds were found) goes into `docs/status/phase-2.md` in task 12.
- `test_at_least_one_threshold_strict`: same run; assert `count ≥ 1` **without** xfail — the design must
  find at least the strongest statutory jump or the backtest genuinely fails.

**Commit:** `test: Egger-Koethenbuerger multi-threshold backtest (stretch goal per spec)`

---

## Task 12 — CLI surface, README, phase status, final green cycle

**Modify:** `src/natex/cli.py`, `tests/test_cli.py`, `README.md`, `.gitignore` (add `benchmarks/out/`).
**Create:** `docs/status/phase-2.md`.

CLI additions (typer, exact):
- `natex datasets [--root PATH]` — table of the 5 registry entries: name, found/missing, rows, ok, and
  the fetch-instruction string for missing ones. Exit code 0 always (informational).
- `natex discover` gains `--degree INT` (default 1), `--coarse/--no-coarse` (default no-coarse),
  `--n-coarse INT` (default 2000). With `--coarse`, use `coarse_to_fine_scan` and include
  `{"coarse": {"frac_centers_scanned": ..., **params}}` in `results.json` (the §6b coverage contract);
  validation/estimation operate on the fine-stage result exactly as before.

**Tests first (extend `tests/test_cli.py`, `CliRunner`):**
- `test_datasets_lists_all_with_fake_root`: tmp root containing only the fake test_score CSV → output has
  5 lines, one "found", four "missing" with source text; exit code 0.
- `test_discover_coarse_smoke`: small synthetic CSV (n=800, planted boundary), `--coarse --n-coarse 200
  --q 9`; results.json contains the `coarse` block with `0 < frac_centers_scanned <= 1` and finite
  `scan.p_value`.
- `test_discover_degree_passthrough`: `--degree 2` runs and records `"degree": 2` in params.

README: extend "Backtests on real data" with the five datasets, their known truths, expected `NATEX_DATA`
layout, and per-dataset fetch instructions (registry `source` strings); add a "Benchmarks" section
(`python benchmarks/run_nig_curve.py --kind real`); flip roadmap row 2 to done.

`docs/status/phase-2.md` (short written status, spec §9): what passed, backtest wall-clock times, the
Egger outcome (thresholds found or documented failure), benchmark curve summary numbers (NIG/power at the
endpoints), and deviations (e.g. LSO randomization on a subsample; ED estimand caveat).

Final gate before the closing commit:
1. `uv run ruff check src tests` clean.
2. `uv run pytest -q` green (no backtests; includes the new CI-small benchmark tests; keep added CI time
   under ~2 min).
3. `NATEX_DATA="/Users/haukehillebrandt/Library/CloudStorage/GoogleDrive-hauke.hillebrandt@gmail.com/My Drive/Gdrive sync/RDD/data" uv run pytest tests/backtests -m backtest -q` — all §8 RDD rows pass (Egger per its
   documented stretch contract).
4. Record (2) and (3) outputs in `docs/status/phase-2.md`.

**Commit:** `feat: datasets/coarse CLI, benchmark docs, phase-2 status (all RDD backtest rows green)`

---

## Dependency order

1 (registry) and 2 (NaN-y estimators) are independent; 3 (vectorized Bernoulli) → 4 (geometry) → 5
(coarse-to-fine) are sequential; 6 (DGP fidelity) → 7 (benchmarks); 8 (clustering) before 9/11; backtests:
9 (ED/inpatient; needs 1, 8) → 10 (LSO; needs 1–5) → 11 (Egger; needs 1, 5, 8); 12 last. Suggested
execution order = task numbering.

## Explicit non-goals (phase 2)

No SuDDDS/DiD (phase 3), no DEE (phase 4), no IV/SC (phase 5), no LLM anything (phase 6), no report/paper
(phase 7). No changepoint-baseline benchmark (KDD Fig 9). No auto-downloading fetch scripts (archive
logins; instructions only). No ANN/approximate neighbors (audit: opt-in later, replicas must share the
approximation). No parallelism across neighborhoods yet — vectorization + coarse-to-fine must meet the
runtime budgets first; add multiprocessing only if task 10's budget fails, as a separate reviewed commit.

## Audit corrections tracked in this phase (map to `docs/math_audit_final.md`)

- Item 1 (+1-rank Monte Carlo p, never "exact") — benchmark `power` definition, all backtest language.
- Item 6 (density = falsification only, frozen geometry) — ED negative-control test.
- Item 10 (first-stage relevance checked, not inferred from LLR) — every backtest asserts
  `weak_instrument` explicitly.
- Items 21/22 (boundary suprema; bracketed Newton) — preserved by the vectorized kernel with parity tests.
- Item 20 (legacy outputs are not ground truth) — backtests assert dataset truths and paper-qualitative
  results only; no legacy parity fixtures.
- §3 adopted improvements: geometry caching across replicas, complement-dedup (already in), exact
  Kmax-NN prefix reuse — task 4.
- Binary-DGP log-odds shift (printed-typo correction, phase 1) re-verified by task 6's
  `test_defaults_unchanged`.
- House rule "NaN never 0.0" — task 2 is its estimator-level enforcement.
