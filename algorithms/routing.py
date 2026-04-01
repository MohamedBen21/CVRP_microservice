# ─────────────────────────────────────────────────────────────────────────────
#  algorithms/routing.py
#  Stop ordering for each vehicle cluster.
#  Mirrors the logic in tsp.util.ts exactly.
#
#  Two algorithms:
#    1. nearest_neighbour  — O(n²), fast
#    2. two_opt            — O(n²) improvement, ~5% better routes
#
#  Combined in optimised_route() for production use.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from utils.haversine import haversine_km, estimated_drive_minutes

Coords = tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class StopPoint:
    __slots__ = ("id", "coords", "package_ids", "meta")

    def __init__(
        self,
        stop_id: str,
        coords: Coords,
        package_ids: list[str],
        meta: dict | None = None,
    ):
        self.id          = stop_id
        self.coords      = coords
        self.package_ids = package_ids
        self.meta        = meta or {}


class RouteResult:
    __slots__ = ("ordered_stops", "total_distance_km", "segment_distances",
                 "segment_drive_minutes", "distance_source")

    def __init__(
        self,
        ordered_stops: list[StopPoint],
        total_distance_km: float,
        segment_distances: list[float],
        segment_drive_minutes: list[int],
        distance_source: str = "haversine",
    ):
        self.ordered_stops          = ordered_stops
        self.total_distance_km      = total_distance_km
        self.segment_distances      = segment_distances
        self.segment_drive_minutes  = segment_drive_minutes
        self.distance_source        = distance_source


# ─────────────────────────────────────────────────────────────────────────────
#  NEAREST NEIGHBOUR
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbour(
    origin: Coords,
    stops: list[StopPoint],
    route_type: str = "local_delivery",
    dist_matrix: list[list[float]] | None = None,
    stop_index_map: dict[str, int] | None = None,
) -> RouteResult:
    """
    Orders stops by nearest-neighbour heuristic.

    dist_matrix / stop_index_map: optional pre-computed distance matrix
    from OSRM.  If provided, matrix[i][j] = road km between stop i and j
    (indexed by stop_index_map[stop.id]).  Falls back to Haversine per leg.
    """
    if not stops:
        return RouteResult([], 0.0, [], [], "haversine")

    if len(stops) == 1:
        d = _dist(origin, stops[0].coords, None, None, None, dist_matrix, stop_index_map)
        return RouteResult(
            stops, d, [d], [estimated_drive_minutes(d, route_type)], "haversine"
        )

    remaining = list(stops)
    ordered: list[StopPoint]  = []
    seg_dist: list[float]     = []
    seg_mins: list[int]       = []
    total    = 0.0
    current  = origin
    current_id = "__origin__"

    while remaining:
        best_d   = math.inf
        best_idx = 0

        for i, stop in enumerate(remaining):
            d = _dist(current, stop.coords, current_id, stop.id,
                      stops, dist_matrix, stop_index_map)
            if d < best_d:
                best_d   = d
                best_idx = i

        chosen = remaining.pop(best_idx)
        ordered.append(chosen)
        seg_dist.append(best_d)
        seg_mins.append(estimated_drive_minutes(best_d, route_type))
        total    += best_d
        current   = chosen.coords
        current_id = chosen.id

    source = "osrm" if dist_matrix is not None else "haversine"
    return RouteResult(ordered, total, seg_dist, seg_mins, source)


# ─────────────────────────────────────────────────────────────────────────────
#  2-OPT IMPROVEMENT
# ─────────────────────────────────────────────────────────────────────────────

def two_opt(
    result: RouteResult,
    origin: Coords,
    route_type: str = "local_delivery",
    dist_matrix: list[list[float]] | None = None,
    stop_index_map: dict[str, int] | None = None,
) -> RouteResult:
    """
    2-opt improvement pass.  Mirrors tsp.util.ts twoOpt() exactly.
    Requires at least 4 stops to make meaningful swaps.
    """
    stops = result.ordered_stops
    if len(stops) < 4:
        return result

    stops = list(stops)   # mutable copy
    n = len(stops)

    # Build coordinate list with origin at index 0
    coords_all: list[Coords] = [origin] + [s.coords for s in stops]

    def d(i: int, j: int) -> float:
        si_id = "__origin__" if i == 0 else stops[i - 1].id
        sj_id = "__origin__" if j == 0 else stops[j - 1].id
        ci = coords_all[i]
        cj = coords_all[j]
        return _dist(ci, cj, si_id, sj_id, stops, dist_matrix, stop_index_map)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                j_next = j + 1 if j + 1 < len(coords_all) else 0
                current_cost = d(i - 1, i) + d(j, j_next)
                swap_cost    = d(i - 1, j) + d(i, j_next)

                if swap_cost < current_cost - 1e-10:
                    stops[i - 1 : j] = stops[i - 1 : j][::-1]
                    coords_all = [origin] + [s.coords for s in stops]
                    improved = True

    return _build_result(origin, stops, route_type, dist_matrix, stop_index_map)


# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED: nearest-neighbour + 2-opt
# ─────────────────────────────────────────────────────────────────────────────

def optimised_route(
    origin: Coords,
    stops: list[StopPoint],
    route_type: str = "local_delivery",
    dist_matrix: list[list[float]] | None = None,
    stop_index_map: dict[str, int] | None = None,
) -> RouteResult:
    """
    Production entry point: nearest-neighbour then 2-opt.
    Pass dist_matrix + stop_index_map if you have an OSRM matrix.
    """
    initial = nearest_neighbour(origin, stops, route_type, dist_matrix, stop_index_map)
    return two_opt(initial, origin, route_type, dist_matrix, stop_index_map)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _dist(
    a: Coords,
    b: Coords,
    a_id: str | None,
    b_id: str | None,
    stops: list[StopPoint] | None,
    dist_matrix: list[list[float]] | None,
    stop_index_map: dict[str, int] | None,
) -> float:
    """
    Returns road distance between two points, using the OSRM matrix when
    available, falling back to Haversine.
    """
    if (
        dist_matrix is not None
        and stop_index_map is not None
        and a_id is not None
        and b_id is not None
        and a_id in stop_index_map
        and b_id in stop_index_map
    ):
        return dist_matrix[stop_index_map[a_id]][stop_index_map[b_id]]

    return haversine_km(a, b)


def _build_result(
    origin: Coords,
    stops: list[StopPoint],
    route_type: str,
    dist_matrix: list[list[float]] | None,
    stop_index_map: dict[str, int] | None,
) -> RouteResult:
    """Recomputes segment distances and drive times for a fully ordered stop list."""
    seg_dist: list[float] = []
    seg_mins: list[int]   = []
    total = 0.0
    prev  = origin
    prev_id = "__origin__"

    for stop in stops:
        d = _dist(prev, stop.coords, prev_id, stop.id, stops, dist_matrix, stop_index_map)
        seg_dist.append(d)
        seg_mins.append(estimated_drive_minutes(d, route_type))
        total  += d
        prev    = stop.coords
        prev_id = stop.id

    source = "osrm" if dist_matrix is not None else "haversine"
    return RouteResult(stops, total, seg_dist, seg_mins, source)
