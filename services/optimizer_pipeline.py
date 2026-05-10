# ─────────────────────────────────────────────────────────────────────────────
#  services/optimizer_pipeline.py
#  Full CVRP / hub-routing pipeline for one branch or hub.
#
#  Three distinct optimization passes
#  ────────────────────────────────────
#
#  1. HUB-TO-HUB pass
#     Input:  manifests destined for the other end of the line.
#     Logic:  No optimization needed.  One transporter = one direct leg.
#             We just group manifests by destination hub and pair each group
#             with one worker + vehicle.  No GA, no clustering, no routing.
#
#  2. HUB-TO-BRANCH pass
#     Input:  manifests destined for local branches served by this hub.
#     Logic:  Each stop = one destination branch.  The optimizer decides WHICH
#             vehicle carries WHICH subset of branch-destined manifests (GA on
#             manifest weights/volumes), then orders the branch stops with
#             nearest-neighbour + 2-opt.  This is functionally the same as the
#             old inter_branch transporter pass but the unit is now a manifest
#             (not a package), and the transporter's assignedBranches list
#             constrains which stops are valid.
#
#  3. DELIVERER pass  (unchanged from previous version)
#     Input:  raw packages with customer delivery addresses.
#     Logic:  GA + nearest-neighbour + 2-opt on package coordinates.
#
#  The three passes are independent and share the same vehicle pool (vehicles
#  are marked used after each pass so no vehicle is double-assigned).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import logging
import math
from api.models import (
    OptimizeRequest, OptimizeResponse,
    PackageInput, ManifestInput, VehicleInput, WorkerInput,
    RouteOutput, StopOutput, UnscheduledPackage, UnscheduledManifest,
)
from algorithms.clustering import (
    cluster_deliverer_packages,
    cluster_transporter_packages,
)
from algorithms.genetic_assignment import (
    PackageGA, VehicleGA, AssignmentResult,
    run_genetic_assignment, CAPACITY_BUFFER,
)
from algorithms.routing import StopPoint, optimised_route
from utils.osrm_client import GlobalDistanceMatrix
from utils.haversine import estimated_drive_minutes

logger = logging.getLogger(__name__)

PRIORITY_MAP = {"same_day": 0, "express": 1, "standard": 2, "urgent": 0}

# Dwell times (minutes)
DELIVERER_DWELL   = 8
HUB_BRANCH_DWELL  = 20   # time to unload a manifest bag at a branch
HUB_HUB_DWELL     = 30   # time to hand off manifests at destination hub


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
#  PASS 1 IMPLEMENTATION: HUB-TO-HUB
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
#  PASS 2 IMPLEMENTATION: HUB-TO-BRANCH
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 3/4 SHARED IMPLEMENTATION: LEGACY TRANSPORTER + DELIVERER
#  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_pass(
    packages: list[PackageInput],
    vehicles: list[VehicleInput],
    workers:  list[WorkerInput],
    origin:   tuple[float, float],
    route_type: str,
    dwell_minutes: int,
    used_vehicle_ids: set[str],
    used_worker_ids:  set[str],
) -> tuple[list[RouteOutput], list[UnscheduledPackage], set[str], set[str]]:
    """
    Runs the full CVRP pipeline for one worker type (legacy transporter or deliverer).
    Returns (routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids).
    """

    ga_packages = _to_ga_packages(packages, route_type)
    ga_vehicles = _to_ga_vehicles(vehicles)

    if route_type == "local_delivery":
        coords = [
            p.destination.coordinates if p.destination else origin
            for p in packages
        ]
        _clusters = cluster_deliverer_packages(coords, len(vehicles))
    else:
        branch_ids = [p.destinationBranchId for p in packages]
        _clusters = cluster_transporter_packages(branch_ids)

    if route_type == "local_delivery":
        all_stop_coords: list[tuple[float, float]] = [
            pkg.destination.coordinates if pkg.destination else origin
            for pkg in packages
        ]
        global_dm = GlobalDistanceMatrix.build(origin, all_stop_coords)

        pkg_coord_to_gidx: dict[int, int] = {}
        for pkg_i, pkg in enumerate(packages):
            c = pkg.destination.coordinates if pkg.destination else origin
            idx = global_dm.index_of(c)
            if idx is not None:
                pkg_coord_to_gidx[pkg_i] = idx

        n_pkgs = len(packages)
        ga_dist_matrix: list[list[float]] | None = (
            [
                [
                    global_dm.matrix[pkg_coord_to_gidx.get(i, 0)][pkg_coord_to_gidx.get(j, 0)]
                    for j in range(n_pkgs)
                ]
                for i in range(n_pkgs)
            ]
            if global_dm.source == "osrm"
            else None
        )
    else:
        global_dm      = None
        ga_dist_matrix = None

    is_deliverer = (route_type == "local_delivery")
    assignments, sorted_ga_vehicles = run_genetic_assignment(
        packages=ga_packages,
        vehicles=ga_vehicles,
        origin_coords=origin,
        is_deliverer=is_deliverer,
        dist_matrix=ga_dist_matrix,
    )

    ga_veh_idx_to_input: dict[int, VehicleInput] = {
        i: vehicles[sv.idx]
        for i, sv in enumerate(sorted_ga_vehicles)
    }

    routes:      list[RouteOutput]        = []
    unscheduled: list[UnscheduledPackage] = []

    available_workers = [w for w in workers if w.id not in used_worker_ids]
    newly_used_vehicle_ids: set[str] = set()
    newly_used_worker_ids:  set[str] = set()

    for assignment in assignments:
        if not available_workers:
            for pkg_idx in assignment.package_indices:
                unscheduled.append(UnscheduledPackage(
                    packageId=packages[pkg_idx].id,
                    reason="No workers available",
                ))
            continue

        veh_input  = ga_veh_idx_to_input.get(assignment.vehicle_idx, vehicles[0])
        worker     = available_workers[0]
        pkg_inputs = [packages[i] for i in assignment.package_indices]

        total_w     = sum(p.weight for p in pkg_inputs)
        total_v     = sum(p.volume for p in pkg_inputs)
        has_fragile = any(p.isFragile for p in pkg_inputs)

        cap_ok = (
            total_w <= veh_input.maxWeight * CAPACITY_BUFFER
            and total_v <= veh_input.maxVolume * CAPACITY_BUFFER
            and (not has_fragile or veh_input.supportsFragile)
        )

        if not cap_ok or not pkg_inputs:
            rescue_vehicle = _find_rescue_vehicle(
                vehicles, total_w, total_v, has_fragile,
                used_ids=newly_used_vehicle_ids,
            )
            if rescue_vehicle is not None:
                veh_input = rescue_vehicle
                cap_ok    = True
                logger.info(
                    f"[pipeline] Rescued {len(pkg_inputs)} packages from bad GA "
                    f"assignment onto {rescue_vehicle.registrationNumber}"
                )
            else:
                for p in pkg_inputs:
                    if p.isFragile and not veh_input.supportsFragile:
                        reason = f"Vehicle {veh_input.registrationNumber} does not support fragile packages"
                    elif total_w > veh_input.maxWeight * CAPACITY_BUFFER:
                        reason = f"Exceeds vehicle weight capacity ({total_w:.1f}kg > {veh_input.maxWeight * CAPACITY_BUFFER:.1f}kg)"
                    elif total_v > veh_input.maxVolume * CAPACITY_BUFFER:
                        reason = f"Exceeds vehicle volume capacity ({total_v:.3f}m³ > {veh_input.maxVolume * CAPACITY_BUFFER:.3f}m³)"
                    else:
                        reason = "No compatible vehicle found after optimization"
                    unscheduled.append(UnscheduledPackage(packageId=p.id, reason=reason))
                continue

        stops = _build_stops(pkg_inputs, route_type, origin=origin)
        if not stops:
            for p in pkg_inputs:
                unscheduled.append(UnscheduledPackage(
                    packageId=p.id, reason="Could not build stop points"
                ))
            continue

        sub_matrix:     list[list[float]] | None = None
        stop_index_map: dict[str, int]           = {}

        if global_dm is not None and global_dm.source == "osrm":
            veh_global_indices = [global_dm.index_of(origin)]
            for stop in stops:
                idx = global_dm.index_of(stop.coords)
                veh_global_indices.append(idx if idx is not None else 0)
            sub_matrix = global_dm.slice(veh_global_indices)
            stop_index_map = {"__origin__": 0}
            for si, stop in enumerate(stops):
                stop_index_map[stop.id] = si + 1

        route_result = optimised_route(
            origin=origin,
            stops=stops,
            route_type=route_type,
            dist_matrix=sub_matrix,
            stop_index_map=stop_index_map if sub_matrix else None,
        )

        total_drive = sum(route_result.segment_drive_minutes)
        total_dwell = len(stops) * dwell_minutes
        total_time  = total_drive + total_dwell

        stop_outputs = [
            _build_stop_output(stop, packages, route_type)
            for stop in route_result.ordered_stops
        ]
        all_pkg_ids = [p.id for p in pkg_inputs]

        distance_source = (
            "n/a"
            if route_type == "inter_branch"
            else route_result.distance_source
        )
        distance_km = (
            0.0
            if route_type == "inter_branch"
            else round(route_result.total_distance_km, 2)
        )

        # Map inter_branch → "hub_to_branch" for legacy transporters; otherwise
        # "local_delivery".  (hub_to_hub routes are never created here.)
        output_route_type = (
            "hub_to_branch" if route_type == "inter_branch" else "local_delivery"
        )

        route = RouteOutput(
            vehicleId=veh_input.id,
            workerId=worker.id,
            routeType=output_route_type,
            stops=stop_outputs,
            packageIds=all_pkg_ids,
            manifestIds=[],
            totalWeight=round(total_w, 2),
            totalVolume=round(total_v, 4),
            distanceKm=distance_km,
            estimatedTimeMinutes=total_time,
            distanceSource=distance_source,
        )

        routes.append(route)
        available_workers.pop(0)
        newly_used_vehicle_ids.add(veh_input.id)
        newly_used_worker_ids.add(worker.id)

        logger.debug(
            f"[pipeline] [{route_type}] vehicle={veh_input.registrationNumber} "
            f"worker={worker.id} pkg={len(pkg_inputs)} "
            f"dist={route_result.total_distance_km:.1f}km "
            f"time={total_time}min src={route_result.distance_source}"
        )

    return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_VEHICLE_TYPE_RANK: dict[str, int] = {
    "motorcycle":  0,
    "car":         1,
    "van":         2,
    "small_truck": 3,
    "large_truck": 4,
}


def _pick_vehicle(
    vehicles:   list[VehicleInput],
    manifests:  list[ManifestInput],
    used_ids:   set[str],
) -> VehicleInput | None:
    """
    Returns the smallest vehicle that can carry the full list of manifests.
    Falls back to the largest vehicle when no single vehicle fits.
    """
    total_w = sum(m.totalWeight for m in manifests)
    total_v = sum(m.totalVolume for m in manifests)
    candidates = [
        v for v in vehicles
        if v.id not in used_ids
        and total_w <= v.maxWeight * CAPACITY_BUFFER
        and total_v <= v.maxVolume * CAPACITY_BUFFER
    ]
    if candidates:
        candidates.sort(key=lambda v: (_VEHICLE_TYPE_RANK.get(v.type, 2), v.maxWeight))
        return candidates[0]
    # Fallback: return the largest available vehicle (GA will redistribute)
    available = [v for v in vehicles if v.id not in used_ids]
    if available:
        available.sort(key=lambda v: (_VEHICLE_TYPE_RANK.get(v.type, 2), v.maxWeight), reverse=True)
        return available[0]
    return None


def _pick_vehicle_for_worker(
    vehicles: list[VehicleInput],
    used_ids: set[str],
) -> VehicleInput | None:
    """Picks the next available vehicle (smallest first) for a hub_to_branch worker."""
    available = [v for v in vehicles if v.id not in used_ids]
    if not available:
        return None
    available.sort(key=lambda v: (_VEHICLE_TYPE_RANK.get(v.type, 2), v.maxWeight))
    return available[0]


def _build_manifest_stops(manifests: list[ManifestInput]) -> list[StopPoint]:
    """
    Groups manifests by destination branch into StopPoints.
    One stop = one destination branch.
    """
    groups: dict[str, dict] = {}
    for m in manifests:
        key = m.destinationBranchId
        if key not in groups:
            groups[key] = {
                "coords":     m.destinationCoordinates,
                "manifest_ids": [],
                "meta": {"destinationBranchId": m.destinationBranchId},
            }
        groups[key]["manifest_ids"].append(m.id)

    stops = []
    for branch_id, g in groups.items():
        stops.append(StopPoint(
            stop_id=branch_id,
            coords=g["coords"],
            package_ids=g["manifest_ids"],   # reuse package_ids field as manifest_ids for routing
            meta=g["meta"],
        ))
    return stops


def _to_ga_packages(packages: list[PackageInput], route_type: str) -> list[PackageGA]:
    result = []
    for i, p in enumerate(packages):
        coords = (
            p.destination.coordinates if (route_type == "local_delivery" and p.destination)
            else None
        )
        result.append(PackageGA(
            idx=i,
            weight=p.weight,
            volume=p.volume,
            is_fragile=p.isFragile,
            coords=coords,
            priority=PRIORITY_MAP.get(p.deliveryPriority, 2),
        ))
    return result


def _to_ga_vehicles(vehicles: list[VehicleInput]) -> list[VehicleGA]:
    return [
        VehicleGA(
            idx=i,
            max_weight=v.maxWeight,
            max_volume=v.maxVolume,
            supports_fragile=v.supportsFragile,
            type_rank=_VEHICLE_TYPE_RANK.get(v.type, 2),
        )
        for i, v in enumerate(vehicles)
    ]


def _build_stops(
    packages:   list[PackageInput],
    route_type: str,
    origin:     tuple[float, float] | None = None,
) -> list[StopPoint]:
    """Groups packages by delivery location into StopPoints."""
    groups: dict[str, dict] = {}

    for pkg in packages:
        if route_type == "local_delivery":
            if not pkg.destination or not pkg.destination.coordinates:
                continue
            lng, lat = pkg.destination.coordinates
            key = f"{lng:.5f},{lat:.5f}"
            if key not in groups:
                groups[key] = {
                    "coords": (lng, lat),
                    "pkg_ids": [],
                    "meta": {
                        "address":       pkg.destination.address,
                        "recipientName": pkg.destination.recipientName,
                    },
                    "destination_branch_id": None,
                }
            groups[key]["pkg_ids"].append(pkg.id)
        else:  # inter_branch (legacy transporter)
            if not pkg.destinationBranchId:
                continue
            key = pkg.destinationBranchId
            if key not in groups:
                placeholder = origin if origin else (0.0, 0.0)
                groups[key] = {
                    "coords": placeholder,
                    "pkg_ids": [],
                    "meta": {"destinationBranchId": pkg.destinationBranchId},
                    "destination_branch_id": pkg.destinationBranchId,
                }
            groups[key]["pkg_ids"].append(pkg.id)

    stops = []
    for key, g in groups.items():
        stops.append(StopPoint(
            stop_id=key,
            coords=g["coords"],
            package_ids=g["pkg_ids"],
            meta=g["meta"],
        ))
    return stops


def _find_rescue_vehicle(
    vehicles:    list[VehicleInput],
    total_w:     float,
    total_v:     float,
    has_fragile: bool,
    used_ids:    set[str],
) -> VehicleInput | None:
    candidates = [
        v for v in vehicles
        if v.id not in used_ids
        and (not has_fragile or v.supportsFragile)
        and total_w <= v.maxWeight * CAPACITY_BUFFER
        and total_v <= v.maxVolume * CAPACITY_BUFFER
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda v: (_VEHICLE_TYPE_RANK.get(v.type, 2), v.maxWeight))
    return candidates[0]


def _build_stop_output(
    stop:       StopPoint,
    packages:   list[PackageInput],
    route_type: str,
) -> StopOutput:
    if route_type == "local_delivery":
        return StopOutput(
            coordinates=stop.coords,
            packageIds=stop.package_ids,
            address=stop.meta.get("address", ""),
            recipientName=stop.meta.get("recipientName", ""),
        )
    else:
        return StopOutput(
            coordinates=stop.coords,
            packageIds=stop.package_ids,
            destinationBranchId=stop.meta.get("destinationBranchId", stop.id),
        )