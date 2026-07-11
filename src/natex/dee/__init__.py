"""DEE layer (Jakubowski et al., JMLR 2023, audit-repaired): disjoint
quasi-experiment repair, local effects, and GP debiasing of observational CATE."""

from natex.dee.vknn import (
    QuasiExperiment,
    VKNNResult,
    experiment_radius,
    select_m_prime,
    voronoi_knn_repair,
)

__all__ = [
    "QuasiExperiment",
    "VKNNResult",
    "experiment_radius",
    "select_m_prime",
    "voronoi_knn_repair",
]
