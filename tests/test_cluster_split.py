from amp_ctmc_2027.data.splits import assign_cluster_splits


def test_cluster_members_never_cross_splits():
    clusters = [f"c{i}" for i in range(30) for _ in range(3)]
    splits = assign_cluster_splits(clusters, seed=8)
    by_cluster = {}
    for cluster, split in zip(clusters, splits):
        by_cluster.setdefault(cluster, set()).add(split)
    assert all(len(values) == 1 for values in by_cluster.values())
