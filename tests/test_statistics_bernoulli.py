import numpy as np
from scipy.special import expit, logit

from natex.scan.statistics import (
    bernoulli_llr_all_splits,
    fit_log_odds_offset,
    offset_log_lik,
)


def test_mle_matches_grid_search():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, size=40)
    eta = logit(p)
    t = rng.binomial(1, expit(eta + 0.7)).astype(float)
    theta = fit_log_odds_offset(t, eta)
    grid = np.linspace(-3, 3, 20001)
    lls = [offset_log_lik(g, t, eta) for g in grid]
    assert abs(theta - grid[int(np.argmax(lls))]) < 2e-3


def test_pure_groups_get_boundary_supremum():
    eta = logit(np.array([0.3, 0.5, 0.7]))
    all_ones = np.ones(3)
    assert fit_log_odds_offset(all_ones, eta) == np.inf
    assert offset_log_lik(np.inf, all_ones, eta) == 0.0  # sup log-lik, not NA
    all_zero = np.zeros(3)
    assert fit_log_odds_offset(all_zero, eta) == -np.inf
    assert offset_log_lik(-np.inf, all_zero, eta) == 0.0


def test_llr_nonnegative_and_sharp_split_scores_high():
    rng = np.random.default_rng(1)
    p = np.full(20, 0.5)
    eta = logit(p)
    t = np.r_[np.ones(10), np.zeros(10)]  # perfectly sharp
    G_sharp = np.r_[np.ones(10, bool), np.zeros(10, bool)][:, None]
    G_rand = rng.random((20, 6)) < 0.5
    llr_sharp = bernoulli_llr_all_splits(t, eta, G_sharp)
    llr_rand = bernoulli_llr_all_splits(t, eta, np.asarray(G_rand))
    assert np.all(llr_rand >= -1e-9)
    assert llr_sharp[0] >= llr_rand.max()  # the sharp RDD must be scoreable and win
