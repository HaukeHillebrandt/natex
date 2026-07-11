"""DEE layer (Jakubowski et al., JMLR 2023, audit-repaired): disjoint
quasi-experiment repair, local effects, and GP debiasing of observational CATE."""

from natex.dee.gp import GPPosterior, HeteroskedasticGP, rbf_kernel, sample_gp_prior
from natex.dee.noise import log_se2_bias, log_se2_measurement_var, smooth_noise
from natex.dee.vknn import (
    QuasiExperiment,
    VKNNResult,
    balance_filter,
    experiment_effects,
    experiment_radius,
    select_m_prime,
    voronoi_knn_repair,
)

__all__ = [
    "GPPosterior",
    "HeteroskedasticGP",
    "QuasiExperiment",
    "VKNNResult",
    "balance_filter",
    "experiment_effects",
    "experiment_radius",
    "log_se2_bias",
    "log_se2_measurement_var",
    "rbf_kernel",
    "sample_gp_prior",
    "select_m_prime",
    "smooth_noise",
    "voronoi_knn_repair",
]
