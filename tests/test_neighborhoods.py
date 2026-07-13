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


def test_issue_19_k_bounds_validated():
    """Issue #19: cKDTree.query(z, k) with k > n fills missing neighbors with
    the sentinel index n, which later crashes the sweep with an uncaught
    IndexError at r[idx]. knn_indices must validate 2 <= k <= n up front with
    a ValueError (which discover() isolates as status="failed"); k == n stays
    legal as an explicit whole-dataset neighborhood."""
    import pytest

    z = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match=r"2 <= k <= n"):
        knn_indices(z, k=11)
    with pytest.raises(ValueError, match=r"2 <= k <= n"):
        knn_indices(z, k=1)
    idx = knn_indices(z, k=10)  # k == n is an explicit, valid choice
    assert idx.shape == (10, 10)
    np.testing.assert_array_equal(idx[:, 0], np.arange(10))
    assert (idx < 10).all()  # no sentinel index n


def _assert_self_first_once(idx):
    for i in range(idx.shape[0]):
        assert idx[i, 0] == i, f"row {i} does not start with self: {idx[i]}"
        assert int(np.sum(idx[i] == i)) == 1, f"self not exactly once in row {i}: {idx[i]}"


def test_issue_24_self_membership_with_exact_duplicates():
    """Issue #24: with > k exact duplicates cKDTree's k-nearest tied subset can
    EXCLUDE self entirely; the old repair loop only fixed reordering, never
    omission, violating the documented center-in-group-1 contract (audit item
    20 / geometry 'self stays in column 0'). When self is absent all returned
    neighbors are provably distance-0 duplicates, so inserting self at column 0
    is geometrically neutral. Do NOT expect permutation-equivariant neighbor
    sets -- with > k ties the kNN subset is inherently arbitrary; only the
    self-first contract must hold."""
    # all-duplicate layout
    _assert_self_first_once(knn_indices(np.zeros((8, 1)), 3))
    # mixed layout: 5 duplicates + 3 distinct points
    z = np.vstack([np.zeros((5, 1)), np.array([[1.0], [2.0], [3.0]])])
    _assert_self_first_once(knn_indices(z, 3))
    # contract holds under row permutations too
    for seed in (0, 1):
        perm = np.random.default_rng(seed).permutation(z.shape[0])
        _assert_self_first_once(knn_indices(z[perm], 3))


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
