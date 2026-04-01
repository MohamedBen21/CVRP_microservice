# ─────────────────────────────────────────────────────────────────────────────
#  utils/osrm_client.py
#  Wraps the OSRM Table API.
#
#  On any error (timeout, connection refused, non-200, bad JSON):
#    → logs a WARNING and returns None
#    → caller falls back to Haversine automatically
#
#  Configuration:
#    OSRM_URL     — base URL, e.g. "http://localhost:5000"
#    OSRM_TIMEOUT — seconds before we give up and fall back
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import logging
import requests
from utils.haversine import Coords, build_distance_matrix, haversine_km

logger = logging.getLogger(__name__)

OSRM_URL     = os.getenv("OSRM_URL", "http://router.project-osrm.org")
OSRM_TIMEOUT = float(os.getenv("OSRM_TIMEOUT", "8"))


def _coords_to_osrm(coords: list[Coords]) -> str:
    """Format [lng, lat] list as 'lng,lat;lng,lat;...' for OSRM URLs."""
    return ";".join(f"{lng},{lat}" for lng, lat in coords)


def fetch_distance_matrix(coords: list[Coords]) -> tuple[list[list[float]], str]:
    """
    Fetches an N×N road-distance matrix (km) from the OSRM Table API.

    Returns:
        (matrix, source) where source is "osrm" or "haversine"

    Falls back to Haversine silently on any OSRM failure.
    OSRM returns durations in seconds; we convert to km using the distances
    annotation (available in OSRM ≥ 5.20).  If the server is older and
    doesn't return distances, we derive km from duration × assumed speed.
    """
    if len(coords) == 0:
        return [], "haversine"

    if len(coords) == 1:
        return [[0.0]], "haversine"

    try:
        coord_str = _coords_to_osrm(coords)
        url = (
            f"{OSRM_URL}/table/v1/driving/{coord_str}"
            "?annotations=distance,duration"
        )

        resp = requests.get(url, timeout=OSRM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            raise ValueError(f"OSRM returned code={data.get('code')}")

        # Prefer the distances annotation (metres → km)
        if "distances" in data:
            raw = data["distances"]
            matrix = [
                [cell / 1000.0 for cell in row]   # metres → km
                for row in raw
            ]
            logger.debug(f"[osrm] distance matrix {len(coords)}×{len(coords)} via distances annotation")
            return matrix, "osrm"

        # Older OSRM: derive from durations × speed heuristic
        # Use 50 km/h as neutral average (urban + highway mix)
        if "durations" in data:
            raw = data["durations"]
            matrix = [
                [(cell / 3600.0) * 50.0 for cell in row]  # seconds → hours → km
                for row in raw
            ]
            logger.debug(f"[osrm] distance matrix {len(coords)}×{len(coords)} via durations fallback")
            return matrix, "osrm"

        raise ValueError("OSRM response missing both 'distances' and 'durations'")

    except requests.exceptions.Timeout:
        logger.warning(f"[osrm] Timeout after {OSRM_TIMEOUT}s — falling back to Haversine")
    except requests.exceptions.ConnectionError:
        logger.warning(f"[osrm] Cannot connect to {OSRM_URL} — falling back to Haversine")
    except Exception as exc:
        logger.warning(f"[osrm] Failed ({exc}) — falling back to Haversine")

    return build_distance_matrix(coords), "haversine"


def fetch_route_distance(origin: Coords, stops: list[Coords]) -> tuple[float, str]:
    """
    Returns total road distance (km) for a fixed ordered sequence of stops
    (origin → stop[0] → stop[1] → ... → stop[-1]) using the OSRM Route API.

    Falls back to summing Haversine legs on any error.
    Useful for final route distance annotation after ordering is settled.
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

        distance_m = data["routes"][0]["distance"]
        return distance_m / 1000.0, "osrm"

    except Exception as exc:
        logger.warning(f"[osrm] route call failed ({exc}) — using Haversine sum")

    # Haversine leg sum
    total = sum(
        haversine_km(all_points[i], all_points[i + 1])
        for i in range(len(all_points) - 1)
    )
    return total, "haversine"
