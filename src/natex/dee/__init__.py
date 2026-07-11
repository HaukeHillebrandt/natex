"""DEE layer (Jakubowski et al., JMLR 2023, audit-repaired): disjoint
quasi-experiment repair, local effects, and GP debiasing of observational CATE."""

from natex.dee.bma import (
    MixturePosterior,
    ModelWeights,
    buffered_folds,
    buffered_stacking_weights,
    loo_weights,
    mixture_posterior,
    mll_weights,
)
from natex.dee.gp import GPPosterior, HeteroskedasticGP, rbf_kernel, sample_gp_prior
from natex.dee.noise import log_se2_bias, log_se2_measurement_var, smooth_noise
from natex.dee.observational import (
    ObservationalEstimator,
    TLearner,
    default_factory,
    experiment_crossfit_cate,
)
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
    "MixturePosterior",
    "ModelWeights",
    "ObservationalEstimator",
    "QuasiExperiment",
    "TLearner",
    "VKNNResult",
    "balance_filter",
    "buffered_folds",
    "buffered_stacking_weights",
    "default_factory",
    "experiment_crossfit_cate",
    "experiment_effects",
    "experiment_radius",
    "log_se2_bias",
    "log_se2_measurement_var",
    "loo_weights",
    "mixture_posterior",
    "mll_weights",
    "rbf_kernel",
    "sample_gp_prior",
    "select_m_prime",
    "smooth_noise",
    "voronoi_knn_repair",
]
