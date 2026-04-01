# ─────────────────────────────────────────────────────────────────────────────
#  algorithms/clustering.py
#  Geographic clustering of packages before GA assignment.
#
#  Why cluster first?
#    • A GA operating on 200 packages across 10 vehicles has a huge search
#      space.  Pre-clustering reduces it by ensuring nearby packages are
#      considered for the same vehicle, which makes GA fitness evals faster
#      and more stable.
#    • For transporter packages (branch → branch), clustering is done by
#      destination branch — packages going to the same branch are trivially
#      grouped.  No ML needed.
#    • For deliverer packages (branch → customer homes), we use K-Means
#      geographic clustering, one cluster per available vehicle.
#
#  Output: list of clusters, each cluster = list of package indices.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import math
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings

logger = logging.getLogger(__name__)

MAX_CLUSTER_SIZE = int(os.getenv("MAX_CLUSTER_SIZE", "25"))


# ─────────────────────────────────────────────────────────────────────────────
#  DELIVERER CLUSTERING  (K-Means geographic)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_deliverer_packages(
    coords: list[tuple[float, float]],   # [lng, lat] per package
    n_vehicles: int,
) -> list[list[int]]:
    """
    Groups packages into at most `n_vehicles` geographic clusters using K-Means.
    Each cluster index list maps back to the original package list.

    Edge cases:
      • 0 packages → []
      • packages ≤ n_vehicles → one cluster per package (trivial)
      • scikit-learn convergence warning silenced (we don't need perfect K-Means)
    """
    n = len(coords)
    if n == 0:
        return []

    k = min(n_vehicles, n)

    if k == 1:
        return [list(range(n))]

    # K-Means operates on (lat, lng) — lat is the primary geographic axis
    # for Algeria (mostly varies N↔S), so we don't need to swap axes.
    X = np.array([[lat, lng] for lng, lat in coords])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result = list(clusters.values())

    # Split any cluster that is too large to be served by one vehicle
    final: list[list[int]] = []
    for cluster in result:
        if len(cluster) <= MAX_CLUSTER_SIZE:
            final.append(cluster)
        else:
            final.extend(_split_cluster(cluster, coords))

    logger.debug(f"[clustering] {n} packages → {len(final)} clusters (k={k})")
    return final


def _split_cluster(cluster: list[int], coords: list[tuple[float, float]]) -> list[list[int]]:
    """Splits an oversized cluster into chunks of MAX_CLUSTER_SIZE."""
    chunks = []
    for i in range(0, len(cluster), MAX_CLUSTER_SIZE):
        chunks.append(cluster[i : i + MAX_CLUSTER_SIZE])
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSPORTER CLUSTERING  (by destination branch — no ML needed)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_transporter_packages(
    destination_branch_ids: list[str | None],
) -> list[list[int]]:
    """
    Groups transporter packages by their destination branch ID.
    Packages with no destination are placed in a single 'unknown' group.

    Returns a list of clusters (each cluster = list of package indices).
    All packages going to the same branch form one cluster — the GA then
    decides which vehicle carries each cluster.
    """
    groups: dict[str, list[int]] = {}
    for idx, branch_id in enumerate(destination_branch_ids):
        key = branch_id if branch_id else "__unknown__"
        groups.setdefault(key, []).append(idx)

    result = list(groups.values())
    logger.debug(f"[clustering] transporter: {len(destination_branch_ids)} packages → {len(result)} branch clusters")
    return result
