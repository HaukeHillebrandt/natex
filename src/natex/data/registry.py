"""Registry and loaders for the five RDD benchmark datasets, keyed on env NATEX_DATA.

Datasets are never committed; ``DatasetInfo.source`` carries human fetch
instructions for reconstructing the local data root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from natex.data.spec import Dataset, DatasetSpec


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    relpath: str  # main CSV relative to the data root
    glob_fallback: str | None  # e.g. "EggerKoethenbuerger_AEJ_Data*.csv" (handles " (1)" suffix)
    treatment: str
    outcome: str | None  # default outcome; loaders accept an override
    forcing: tuple[str, ...]
    covariates: tuple[str, ...]  # explicit, never "auto" (reproducibility)
    n_rows: int | None  # expected data rows; None = don't check
    source: str  # human fetch instructions (URL + landing page)
    notes: str = ""


@dataclass(frozen=True)
class DatasetStatus:
    name: str
    found: bool
    path: Path | None
    n_rows: int | None
    ok: bool
    message: str


_LSO_COVARIATES = (
    "dist_from_cut",
    "hsgrade_pct",
    "totcredits_year1",
    "age_at_entry",
    "sex",
    "bpl_north_america",
    "mtongue",
    "loc_campus1",
    "loc_campus2",
)

REGISTRY: dict[str, DatasetInfo] = {
    "test_score_2012": DatasetInfo(
        name="test_score_2012",
        relpath="test_score_2012/RDD_Guide_Dataset_0.csv",
        glob_fallback=None,
        treatment="treat",
        outcome="posttest",
        forcing=("age", "pretest"),
        covariates=(
            "gender",
            "sped",
            "frlunch",
            "esol",
            "black",
            "white",
            "hispanic",
            "asian",
            "age",
            "pretest",
        ),
        n_rows=2767,
        source=(
            "MDRC RDD practice dataset from Jacob, Zhu, Somers & Bloom (2012), "
            "'A Practical Guide to Regression Discontinuity'. Download from "
            "https://www.mdrc.org/publication/practical-guide-regression-discontinuity "
            "and place the CSV at test_score_2012/RDD_Guide_Dataset_0.csv under NATEX_DATA."
        ),
        notes=(
            "Perfectly sharp RDD: treat == (pretest < 215); treatment goes to LOW "
            "scorers. The file has 2767 data rows and no trailing newline (wc -l "
            "style counts report 2766); Dataset keeps 2606 after scan-column NaN drop."
        ),
    ),
    "academic_probation": DatasetInfo(
        name="academic_probation",
        relpath="AcademicProbation_LSO_2010/data_orig.csv",
        glob_fallback=None,
        treatment="probation_year1",
        outcome="GPA_year2",
        forcing=("dist_from_cut", "hsgrade_pct", "totcredits_year1", "age_at_entry"),
        covariates=_LSO_COVARIATES,
        n_rows=44362,
        source=(
            "Lindo, Sanders & Oreopoulos (2010), 'Ability, Gender, and Performance "
            "Standards: Evidence from Academic Probation', AEJ: Applied 2(2). Data "
            "archive on openICPSR (login-gated): search the AEJ:Applied data archives "
            "at https://www.openicpsr.org/ and place data_orig.csv at "
            "AcademicProbation_LSO_2010/data_orig.csv under NATEX_DATA."
        ),
        notes=(
            "Fuzzy RDD at dist_from_cut = 0. Outcomes 13-56% missing (later cohorts "
            "censored); sex/mtongue are string categoricals (Dataset one-hot-encodes)."
        ),
    ),
    "ed_visits": DatasetInfo(
        name="ed_visits",
        relpath="ED_visits/P03_ED_Analysis_File.csv",
        glob_fallback=None,
        treatment="priv_all",
        outcome="all",
        forcing=("months_23",),
        covariates=("months_23",),
        n_rows=161,
        source=(
            "Anderson, Dobkin & Gross (2012), 'The Effect of Health Insurance on "
            "Emergency Department Visits', AEJ: Economic Policy 4(1). Data archive on "
            "openICPSR (login-gated): search the AEJ data archives at "
            "https://www.openicpsr.org/ and place P03_ED_Analysis_File.csv at "
            "ED_visits/P03_ED_Analysis_File.csv under NATEX_DATA."
        ),
        notes=(
            "161 aggregated age-in-months cells (not individuals); fuzzy RDDs at ages "
            "19 and 23 (months_23 = -48 and 0); treatment is a continuous insured share."
        ),
    ),
    "inpatient_visits": DatasetInfo(
        name="inpatient_visits",
        relpath="Inpatient_visits/P10_Inpatient_CSV_File.csv",
        glob_fallback=None,
        treatment="TOT_priv_ALL",
        outcome="TOT_ALL",
        forcing=("months_23",),
        covariates=("months_23",),
        n_rows=73,
        source=(
            "Anderson, Dobkin & Gross (2012) inpatient companion file, AEJ: Economic "
            "Policy data archive on openICPSR (login-gated): search the AEJ data "
            "archives at https://www.openicpsr.org/ and place P10_Inpatient_CSV_File.csv "
            "at Inpatient_visits/P10_Inpatient_CSV_File.csv under NATEX_DATA."
        ),
        notes="Only 73 aggregated cells; small-n robustness check at the age-23 cutoff.",
    ),
    "egger_koethenbuerger": DatasetInfo(
        name="egger_koethenbuerger",
        relpath="EggerKoethenbuerger_AEJ_Data.csv",
        glob_fallback="EggerKoethenbuerger_AEJ_Data*.csv",
        treatment="rcsize",
        outcome="exptot",
        forcing=("log_pop",),
        covariates=("log_pop",),
        n_rows=43175,
        source=(
            "Egger & Koethenbuerger (2010), 'Government Spending and Legislative "
            "Organization: Quasi-experimental Evidence from Germany', AEJ: Applied "
            "2(4). Data archive on openICPSR (login-gated): search the AEJ:Applied "
            "data archives at https://www.openicpsr.org/ and place the CSV at "
            "EggerKoethenbuerger_AEJ_Data.csv under NATEX_DATA (a downloaded "
            "'EggerKoethenbuerger_AEJ_Data (1).csv' filename also works)."
        ),
        notes=(
            "Bavarian municipality panel; council size (rcsize) jumps at statutory "
            "population thresholds. Forcing variable log_pop = log(wpop) is derived at "
            "load time; rows with missing/nonpositive wpop are dropped."
        ),
    ),
}


def _info(name: str) -> DatasetInfo:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}") from None


def data_root(root: str | Path | None = None) -> Path:
    """Resolve the benchmark data root: explicit arg wins, else env NATEX_DATA."""
    if root is not None:
        return Path(root)
    env = os.environ.get("NATEX_DATA")
    if not env:
        raise RuntimeError(
            "NATEX_DATA is not set and no root was passed. Point NATEX_DATA at the "
            "benchmark data root containing test_score_2012/, "
            "AcademicProbation_LSO_2010/, ED_visits/, Inpatient_visits/ and "
            "EggerKoethenbuerger_AEJ_Data.csv (see DatasetInfo.source for how to "
            "obtain each file)."
        )
    return Path(env)


def locate(name: str, root: str | Path | None = None) -> Path:
    """Resolve the CSV path for a registered dataset, using the glob fallback if set."""
    info = _info(name)
    base = data_root(root)
    path = base / info.relpath
    if path.is_file():
        return path
    if info.glob_fallback is not None:
        matches = sorted(p for p in path.parent.glob(info.glob_fallback) if p.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"dataset {name!r}: no file at {path}"
        + (f" (glob {info.glob_fallback!r} matched nothing)" if info.glob_fallback else "")
        + f". How to obtain it: {info.source}"
    )


def verify(name: str, root: str | Path | None = None) -> DatasetStatus:
    """Check that a registered dataset is present with the expected row count."""
    info = _info(name)
    try:
        path = locate(name, root)
    except (FileNotFoundError, RuntimeError) as exc:
        return DatasetStatus(
            name=name, found=False, path=None, n_rows=None, ok=False, message=str(exc)
        )
    n = int(len(pd.read_csv(path, usecols=[0])))
    if info.n_rows is not None and n != info.n_rows:
        return DatasetStatus(
            name=name,
            found=True,
            path=path,
            n_rows=n,
            ok=False,
            message=f"dataset {name!r}: {path} has {n} data rows, expected {info.n_rows}",
        )
    return DatasetStatus(name=name, found=True, path=path, n_rows=n, ok=True, message="ok")


def _prepare(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-specific derived columns (documented in REGISTRY[name].notes)."""
    if name == "egger_koethenbuerger":
        # Population is heavily right-skewed and statutory thresholds are
        # multiplicative: scan on log_pop. log() needs strictly positive wpop.
        df = df[df["wpop"].notna() & (df["wpop"] > 0)].reset_index(drop=True)
        df = df.assign(log_pop=np.log(df["wpop"].to_numpy(dtype=float)))
    return df


def load_dataset(
    name: str, root: str | Path | None = None, outcome: str | None = "default"
) -> Dataset:
    """Load a registered dataset as a natex Dataset.

    ``outcome="default"`` uses ``DatasetInfo.outcome``; ``None`` loads without an
    outcome (pure discovery); any other string overrides the default column.
    """
    info = _info(name)
    path = locate(name, root)
    df = _prepare(name, pd.read_csv(path))
    out = info.outcome if outcome == "default" else outcome
    spec = DatasetSpec(
        treatment=info.treatment,
        outcome=out,
        forcing=list(info.forcing),
        covariates=list(info.covariates),
    )
    return Dataset(df, spec)
