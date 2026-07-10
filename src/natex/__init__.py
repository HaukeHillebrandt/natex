"""natex: automated natural-experiment discovery."""

from natex.data.spec import Dataset, DatasetSpec
from natex.rdd.lord3 import LoRD3Result, lord3_scan

__version__ = "0.1.0.dev0"
__all__ = ["Dataset", "DatasetSpec", "LoRD3Result", "lord3_scan", "__version__"]
