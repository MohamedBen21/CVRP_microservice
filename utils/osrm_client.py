# ─────────────────────────────────────────────────────────────────────────────
#  utils/osrm_client.py
#  OSRM Table API client — ONE call per optimization pass, not one per vehicle.
#
#  Architecture
#  ────────────
#  Old (broken) flow:
#    for each vehicle:
#        fetch_distance_matrix(vehicle_stops)   ← N sequential HTTP calls
#
#  New (correct) flow:
#    all_coords = origin + every package coordinate (deduplicated)
#    global_matrix, source = fetch_global_matrix(all_coords)   ← 1 HTTP call
#    for each vehicle:
#        sub = slice_matrix(global_matrix, indices_for_this_vehicle)
#
#  Why the old flow hit the timeout
#  ─────────────────────────────────
#  Docker OSRM is single-threaded per request.  3 sequential 8-second timeouts
#  = 24 s minimum; with startup overhead the total hit 28 s.  One call with all
#  coordinates takes ~200 ms regardless of point count for small datasets.
#
#  Configuration  (all via .env or environment)
#  ─────────────────────────────────────────────
#  OSRM_URL             base URL of your OSRM instance
#  OSRM_TIMEOUT         seconds for the single global matrix call (default 15)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import logging
import requests
from utils.haversine import Coords, build_distance_matrix, haversine_km

logger = logging.getLogger(__name__)

OSRM_URL     = os.getenv("OSRM_URL",     "http://router.project-osrm.org")
OSRM_TIMEOUT = float(os.getenv("OSRM_TIMEOUT", "15"))   # raised from 8 → 15 for safety


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _coords_to_osrm(coords: list[Coords]) -> str:
    """Format [lng, lat] list as 'lng,lat;lng,lat;...' for OSRM URLs."""
    return ";".join(f"{lng},{lat}" for lng, lat in coords)


def _call_table_api(coords: list[Coords]) -> tuple[list[list[float]], str]:
    """
    Issues ONE OSRM Table API request and returns an N×N distance matrix (km).

    Tries `?annotations=distance` first (OSRM ≥ 5.20).
    Falls back to `?annotations=duration` and derives km via 50 km/h heuristic
    if the server is older.
    Returns ("haversine" source + Haversine matrix) on any error so callers
    never have to handle None.
    """
    if len(coords) <= 1:
        return ([[0.0]] if coords else []), "haversine"

    try:
        coord_str = _coords_to_osrm(coords)
        url = (
            f"{OSRM_URL}/table/v1/driving/{coord_str}"
            "?annotations=distance,duration"
        )
        logger.debug(f"[osrm] table request — {len(coords)} points → {url[:120]}…")

        resp = requests.get(url, timeout=OSRM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise ValueError(f"OSRM code={data.get('code')}")

        # Prefer real distances (metres → km)
        if "distances" in data:
            matrix = [
                [cell / 1000.0 for cell in row]
                for row in data["distances"]
            ]
            logger.debug(
                f"[osrm] {len(coords)}×{len(coords)} matrix via distances annotation"
            )
            return matrix, "osrm"

        # Older server: derive from durations (seconds) at 50 km/h average
        if "durations" in data:
            matrix = [
                [(cell / 3600.0) * 50.0 for cell in row]
                for row in data["durations"]
            ]
            logger.debug(
                f"[osrm] {len(coords)}×{len(coords)} matrix via durations annotation"
            )
            return matrix, "osrm"

        raise ValueError("OSRM response missing both 'distances' and 'durations'")

    except requests.exceptions.Timeout:
        logger.warning(
            f"[osrm] Timeout after {OSRM_TIMEOUT}s ({len(coords)} points) "
            "— falling back to Haversine"
        )
    except requests.exceptions.ConnectionError:
        logger.warning(
            f"[osrm] Cannot connect to {OSRM_URL} — falling back to Haversine"
        )
    except Exception as exc:
        logger.warning(f"[osrm] Table API failed ({exc}) — falling back to Haversine")

    return build_distance_matrix(coords), "haversine"


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — used by the pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GlobalDistanceMatrix:
    """
    Holds the ONE distance matrix computed for the entire optimization pass.

    Usage:
        gdm = GlobalDistanceMatrix.build(origin, all_package_coords)
        sub = gdm.slice(indices)      # indices relative to all_package_coords
        matrix[i][j]                  # distance in km

    The matrix is indexed as:
        index 0        = origin (branch)
        index 1..N     = package / stop coordinates in the order passed to build()

    The `source` attribute is "osrm" or "haversine" — the pipeline propagates
    this into each RouteOutput.distanceSource.
    """

    def __init__(
        self,
        matrix: list[list[float]],
        coords: list[Coords],          # full coord list in matrix order
        source: str,
    ):
        self.matrix  = matrix
        self.coords  = coords
        self.source  = source
        self._coord_to_idx: dict[str, int] = {
            _coord_key(c): i for i, c in enumerate(coords)
        }

    @classmethod
    def build(
        cls,
        origin: Coords,
        stop_coords: list[Coords],    # all unique stop coordinates for this pass
    ) -> "GlobalDistanceMatrix":
        """
        Fires ONE OSRM call covering origin + all stop coordinates.
        Deduplicates coordinates so identical stops don't inflate the matrix.
        """
        # Deduplicate while preserving order (origin always at index 0)
        seen: set[str] = set()
        unique: list[Coords] = []
        for c in [origin] + stop_coords:
            key = _coord_key(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        matrix, source = _call_table_api(unique)
        logger.info(
            f"[osrm] Global matrix built — {len(unique)} unique points, "
            f"source={source}"
        )
        return cls(matrix, unique, source)

    def index_of(self, coord: Coords) -> int | None:
        """Returns the matrix row/col index for a coordinate, or None if not found."""
        return self._coord_to_idx.get(_coord_key(coord))

    def slice(self, indices: list[int]) -> list[list[float]]:
        """
        Extracts a sub-matrix for a given list of global indices.
        Returns an M×M matrix where M = len(indices).
        Used to give each vehicle its own compact distance matrix.
        """
        return [
            [self.matrix[i][j] for j in indices]
            for i in indices
        ]

    def dist(self, a: Coords, b: Coords) -> float:
        """
        Point-to-point distance (km) between two coordinates.
        Falls back to Haversine if either coordinate is not in the matrix.
        """
        ia = self._coord_to_idx.get(_coord_key(a))
        ib = self._coord_to_idx.get(_coord_key(b))
        if ia is None or ib is None:
            return haversine_km(a, b)
        return self.matrix[ia][ib]


def _coord_key(c: Coords) -> str:
    """Stable string key for a coordinate — 6 decimal places ≈ 11 cm precision."""
    return f"{c[0]:.6f},{c[1]:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY SHIM  (keeps old callers working without changes)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_distance_matrix(coords: list[Coords]) -> tuple[list[list[float]], str]:
    """
    Backward-compatible wrapper around _call_table_api.
    Still used by any code that has not been migrated to GlobalDistanceMatrix.
    """
    return _call_table_api(coords)


def fetch_route_distance(origin: Coords, stops: list[Coords]) -> tuple[float, str]:
    """
    Total road distance (km) for an ordered sequence via the OSRM Route API.
    Falls back to Haversine leg sum on any error.
    """
    all_points = [origin] + stops
    if len(all_points) < 2:
        return 0.0, "haversine"

    try:
        coord_str = _coords_to_osrm(all_points)
        url = f"{OSRM_URL}/route/v1/driving/{coord_str}?overview=false"
        resp = requests.get(url, timeout=OSRM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise ValueError(f"OSRM route code={data.get('code')}")

        return data["routes"][0]["distance"] / 1000.0, "osrm"

    except Exception as exc:
        logger.warning(f"[osrm] route call failed ({exc}) — using Haversine sum")

    total = sum(
        haversine_km(all_points[i], all_points[i + 1])
        for i in range(len(all_points) - 1)
    )
    return total, "haversine"