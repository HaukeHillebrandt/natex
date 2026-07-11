"""natex: automated natural-experiment discovery."""

from natex.data.registry import load_dataset
from natex.data.spec import Dataset, DatasetSpec
from natex.data.synthetic_did import make_did_synthetic
from natex.did.effects import did_effect
from natex.did.panel import build_panel
from natex.did.suddds import DiDDiscovery, SuDDDSResult, suddds_scan
from natex.rdd.lord3 import LoRD3Result, lord3_scan
from natex.scan.coarse import CoarseToFineResult, coarse_to_fine_scan

__version__ = "0.1.0.dev0"
__all__ = [
    "CoarseToFineResult",
    "Dataset",
    "DatasetSpec",
    "DiDDiscovery",
    "LoRD3Result",
    "SuDDDSResult",
    "build_panel",
    "coarse_to_fine_scan",
    "did_effect",
    "load_dataset",
    "lord3_scan",
    "make_did_synthetic",
    "suddds_scan",
    "__version__",
]
