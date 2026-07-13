import numpy as np

from natex.scan.neighborhoods import candidate_partitions, knn_indices


def test_knn_self_first():
    z = np.array([[0.0], [1.0], [10.0], [11.0]])
    idx = knn_indices(z, k=2)
    assert idx.shape == (4, 2)
    np.testing.assert_array_equal(idx[:, 0], [0, 1, 2, 3])
    assert idx[0, 1] == 1 and idx[2, 1] == 3


def test_partitions_match_naive_and_center_in_group1():
    rng = np.random.default_rng(0)
    cz = rng.normal(size=(8, 2))
    cz[0] = 0.0  # the center
    G, keep = candidate_partitions(cz)
    # center row must always be group 1 (tie convention)
    assert np.all(G[0, :])
    # each kept column reproduces the naive rule for its normal
    for col, j in enumerate(keep):
        naive = (cz @ cz[j]) >= 0.0
        assert np.array_equal(G[:, col], naive) or np.array_equal(G[:, col], ~naive)


def test_dedup_complements_and_duplicates():
    # Under the tie rule (signed distance >= 0 -> group 1) the tied center is
    # group 1 under BOTH orientations, so the +1 and -1 normals give DISTINCT
    # partitions, not complements: both must survive. Only genuinely identical
    # membership masks (duplicate normals) collapse.
    cz = np.array([[0.0], [1.0], [-1.0]])
    G, keep = candidate_partitions(cz)
    assert G.shape[1] == 2
    cz2 = np.array([[0.0], [1.0], [1.0], [-1.0]])
    G2, _ = candidate_partitions(cz2)
    assert G2.shape[1] == 2


def test_issue_8_antipodal_normals_keep_both_distinct_partitions():
    """Issue #8: the antipodal key (dots <= 0) deduped two genuinely different
    partitions — tied rows (the center at least) are group 1 under both
    orientations, so ~mask never occurs among candidates and the second
    partition (with its own, possibly much larger LLR) was silently lost."""
    cz = np.array([[0.0], [1.0], [-1.0]])
    G, keep = candidate_partitions(cz)
    masks = {tuple(int(v) for v in G[:, j]) for j in range(G.shape[1])}
    assert masks == {(1, 1, 0), (1, 0, 1)}
    assert np.all(G[0, :])  # center stays group 1 in every kept partition
