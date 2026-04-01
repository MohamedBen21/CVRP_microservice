# ─────────────────────────────────────────────────────────────────────────────
#  utils/haversine.py
#  Great-circle distance — mirrors haversine.util.ts exactly.
#  Used as the fallback when OSRM is unavailable.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Sequence

Coords = tuple[float, float]   # (longitude, latitude)

EARTH_RADIUS_KM = 6371.0

AVG_SPEED_KMH: dict[str, float] = {
    "inter_branch":   80.0,
    "local_delivery": 35.0,
}


def haversine_km(a: Coords, b: Coords) -> float:
    """Straight-line distance in km between two [lng, lat] points."""
    lng1, lat1 = a
    lng2, lat2 = b

    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)

    sin_lat = math.sin(d_lat / 2)
    sin_lng = math.sin(d_lng / 2)

    h = (
        sin_lat * sin_lat
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * sin_lng * sin_lng
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def estimated_drive_minutes(distance_km: float, route_type: str) -> int:
    speed = AVG_SPEED_KMH.get(route_type, 50.0)
    return round((distance_km / speed) * 60)


def build_distance_matrix(coords: list[Coords]) -> list[list[float]]:
    """
    N×N symmetric distance matrix using Haversine.
    matrix[i][j] = km from coords[i] to coords[j].
    """
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(coords[i], coords[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix
