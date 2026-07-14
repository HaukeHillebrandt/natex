"""natex: automated natural-experiment discovery."""

from natex.data.registry import load_dataset
from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic_dee import make_dee_synthetic
from natex.data.synthetic_did import make_did_synthetic
from natex.data.synthetic_kink import make_dik_synthetic, make_rkd_synthetic
from natex.dee.debias import DEEResult, dee_debias
from natex.dee.vknn import voronoi_knn_repair
from natex.did.effects import did_effect
from natex.discover import ConfigRecord, DiscoverReport, discover
from natex.did.panel import build_panel
from natex.did.suddds import DiDDiscovery, SuDDDSResult, suddds_scan
from natex.intake.analyst import IntakeReport, study
from natex.intake.plans import DesignCandidate, SearchPlan
from natex.intake.prep import PrepPlan
from natex.kink import (
    KinkEstimate,
    covariate_kinks,
    density_kink_difference,
    difference_in_kinks,
    event_study_kinks,
    placebo_kinks,
    regression_kink,
    sensitivity_grid,
)
from natex.rdd.lord3 import LoRD3Result, lord3_scan
from natex.report.bundle import ResultsBundle
from natex.scan.coarse import CoarseToFineResult, coarse_to_fine_scan

__version__ = "0.1.0"
__all__ = [
    "CoarseToFineResult",
    "ConfigRecord",
    "DEEResult",
    "Dataset",
    "DatasetSpec",
    "DesignCandidate",
    "DiDDiscovery",
    "DiscoverReport",
    "IntakeReport",
    "KinkEstimate",
    "LoRD3Result",
    "PrepPlan",
    "ResultsBundle",
    "SearchPlan",
    "SuDDDSResult",
    "build_panel",
    "coarse_to_fine_scan",
    "covariate_kinks",
    "dee_debias",
    "density_kink_difference",
    "did_effect",
    "difference_in_kinks",
    "discover",
    "event_study_kinks",
    "load_dataset",
    "lord3_scan",
    "make_dee_synthetic",
    "make_dik_synthetic",
    "make_did_synthetic",
    "make_rkd_synthetic",
    "placebo_kinks",
    "regression_kink",
    "sensitivity_grid",
    "study",
    "suddds_scan",
    "voronoi_knn_repair",
    "__version__",
]
