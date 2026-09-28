"""Deterministic cluster-group train/validation/test assignment."""

from __future__ import annotations

import hashlib


def assign_cluster_splits(
    cluster_ids: list[str],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[str]:
    if (
        len(fractions) != 3
        or any(f < 0 for f in fractions)
        or abs(sum(fractions) - 1.0) > 1e-8
    ):
        raise ValueError(
            "split fractions must be three nonnegative values summing to one"
        )
    unique = sorted(
        set(cluster_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    n_train = int(len(unique) * fractions[0])
    n_val = int(len(unique) * fractions[1])
    mapping = {cluster: "train" for cluster in unique[:n_train]}
    mapping.update({cluster: "val" for cluster in unique[n_train : n_train + n_val]})
    mapping.update({cluster: "test" for cluster in unique[n_train + n_val :]})
    result = [mapping[c] for c in cluster_ids]
    groups: dict[str, set[str]] = {}
    for cluster, split in zip(cluster_ids, result):
        groups.setdefault(cluster, set()).add(split)
    if any(len(v) != 1 for v in groups.values()):
        raise AssertionError("a homology cluster crossed splits")
    return result
