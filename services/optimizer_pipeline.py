# ─────────────────────────────────────────────────────────────────────────────
#  services/optimizer_pipeline.py
#  Full CVRP pipeline for one branch.
#
#  Flow:
#    1.  Split packages by type (transporter vs deliverer)
#    2.  Pre-cluster packages geographically
#    3.  Run GA to jointly assign packages → vehicles (capacity-aware)
#    4.  Fetch OSRM distance matrix per vehicle cluster (falls back to Haversine)
#    5.  Build ordered routes (nearest-neighbour + 2-opt per vehicle)
#    6.  Assign workers → vehicles (one worker per vehicle, round-robin)
#    7.  Return OptimizeResponse
#
#  Key design decision: the GA operates on the full set of packages at once,
#  not per-cluster.  Clustering is used only to seed a smarter initial
#  population and to reduce the OSRM matrix size per vehicle.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import logging
import math
from api.models import (
    OptimizeRequest, OptimizeResponse,
    PackageInput, VehicleInput, WorkerInput,
    RouteOutput, StopOutput, UnscheduledPackage,
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

PRIORITY_MAP = {"same_day": 0, "express": 1, "standard": 2}

# Dwell times (minutes) — mirrors the TS constants
DELIVERER_DWELL = 8
TRANSPORTER_DWELL = 20


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(req: OptimizeRequest) -> OptimizeResponse:
    origin = req.branch.coordinates  # [lng, lat]

    # ── Separate packages by type ─────────────────────────────────────────────
    transporter_pkgs = [p for p in req.packages if p.destinationBranchId]
    deliverer_pkgs   = [
        p for p in req.packages
        if not p.destinationBranchId
        and p.deliveryType == "home"
        and p.destination is not None
        and p.destination.coordinates is not None
    ]
    # Packages that can't be routed (no destination info)
    unroutable = [
        p for p in req.packages
        if p not in transporter_pkgs and p not in deliverer_pkgs
    ]

    # Separate worker pools
    transporters = [w for w in req.workers if w.role == "transporter"]
    deliverers   = [w for w in req.workers if w.role == "deliverer"]

    # Separate vehicle pools — same vehicles shared, but we allocate per-pass
    # (mirrors Node.js: vehicles are popped as they get assigned)
    vehicles = req.vehicles

    all_routes:      list[RouteOutput]      = []
    all_unscheduled: list[UnscheduledPackage] = []

    # Unroutable packages
    for p in unroutable:
        all_unscheduled.append(
            UnscheduledPackage(packageId=p.id, reason="Missing destination coordinates")
        )

    used_vehicle_ids: set[str] = set()
    used_worker_ids:  set[str] = set()

    # ── TRANSPORTER PASS ──────────────────────────────────────────────────────
    if transporter_pkgs and transporters:
        available_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]
        t_routes, t_unscheduled, t_used_v, t_used_w = _optimize_pass(
            packages=transporter_pkgs,
            vehicles=available_vehicles,
            workers=transporters,
            origin=origin,
            route_type="inter_branch",
            dwell_minutes=TRANSPORTER_DWELL,
            used_vehicle_ids=used_vehicle_ids,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(t_routes)
        all_unscheduled.extend(t_unscheduled)
        used_vehicle_ids.update(t_used_v)
        used_worker_ids.update(t_used_w)
    else:
        for p in transporter_pkgs:
            reason = (
                "No transporters available" if not transporters
                else "No vehicles available"
            )
            all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason=reason))

    # ── DELIVERER PASS ────────────────────────────────────────────────────────
    if deliverer_pkgs and deliverers:
        available_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]
        d_routes, d_unscheduled, d_used_v, d_used_w = _optimize_pass(
            packages=deliverer_pkgs,
            vehicles=available_vehicles,
            workers=deliverers,
            origin=origin,
            route_type="local_delivery",
            dwell_minutes=DELIVERER_DWELL,
            used_vehicle_ids=used_vehicle_ids,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(d_routes)
        all_unscheduled.extend(d_unscheduled)
        used_vehicle_ids.update(d_used_v)
        used_worker_ids.update(d_used_w)
    else:
        for p in deliverer_pkgs:
            reason = (
                "No deliverers available" if not deliverers
                else "No vehicles available"
            )
            all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason=reason))

    return OptimizeResponse(
        routes=all_routes,
        unscheduled=all_unscheduled,
        meta={
            "totalPackages":    len(req.packages),
            "scheduled":        sum(len(r.packageIds) for r in all_routes),
            "unscheduled":      len(all_unscheduled),
            "routesCreated":    len(all_routes),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PER-PASS OPTIMIZER  (shared by transporter + deliverer)
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
    Runs the full CVRP pipeline for one worker type.

    Returns (routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids).
    """

    # ── Convert to internal GA types ──────────────────────────────────────────
    ga_packages = _to_ga_packages(packages, route_type)
    ga_vehicles = _to_ga_vehicles(vehicles)

    # ── Geographic clustering (seeds smarter GA population) ──────────────────
    if route_type == "local_delivery":
        coords = [
            p.destination.coordinates if p.destination else origin
            for p in packages
        ]
        _clusters = cluster_deliverer_packages(coords, len(vehicles))
    else:
        branch_ids = [p.destinationBranchId for p in packages]
        _clusters = cluster_transporter_packages(branch_ids)

    # ── OSRM: ONE global matrix call per pass (deliverer only) ──────────────
    #
    # Transporter pass: all packages map to `origin` as a placeholder because
    # real branch coordinates are not in the request — Node.js fetches them at
    # persist time.  Calling OSRM with a single deduplicated point is wasteful
    # and always falls back to Haversine anyway, so we skip it entirely.
    #
    # Deliverer pass: collect every unique customer coordinate and build one
    # matrix covering all stops.  Each vehicle's sub-matrix is sliced cheaply
    # in memory after the GA assignment — no further HTTP calls are made.

    if route_type == "local_delivery":
        all_stop_coords: list[tuple[float, float]] = [
            pkg.destination.coordinates if pkg.destination else origin
            for pkg in packages
        ]
        global_dm = GlobalDistanceMatrix.build(origin, all_stop_coords)

        # pkg_idx → global matrix index (for GA fitness function)
        pkg_coord_to_gidx: dict[int, int] = {}
        for pkg_i, pkg in enumerate(packages):
            c = pkg.destination.coordinates if pkg.destination else origin
            idx = global_dm.index_of(c)
            if idx is not None:
                pkg_coord_to_gidx[pkg_i] = idx

        # N×N package-level distance matrix for GA
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
        # Transporter pass — no OSRM call, no distance matrix.
        # GA uses Haversine internally (all coords are origin placeholders,
        # so all distances are 0 — irrelevant for the fitness function since
        # transporter packages have no route cost contribution).
        global_dm    = None
        ga_dist_matrix = None

    # ── Genetic Algorithm: joint package→vehicle assignment ───────────────────
    # is_deliverer: enforces the ≤15 packages/vehicle hard constraint and
    # uses the correct penalty scale inside the fitness function.
    is_deliverer = (route_type == "local_delivery")
    assignments, sorted_ga_vehicles = run_genetic_assignment(
        packages=ga_packages,
        vehicles=ga_vehicles,
        origin_coords=origin,
        is_deliverer=is_deliverer,
        dist_matrix=ga_dist_matrix,
    )

    # Build a lookup from GA vehicle index → original VehicleInput.
    # The GA returns assignments using sorted_ga_vehicles indices (small→large),
    # so we map sorted_ga_vehicles[i].idx → vehicles[original_idx].
    ga_veh_idx_to_input: dict[int, VehicleInput] = {
        i: vehicles[sv.idx]
        for i, sv in enumerate(sorted_ga_vehicles)
    }

    # ── Build routes per vehicle ──────────────────────────────────────────────
    routes:      list[RouteOutput]      = []
    unscheduled: list[UnscheduledPackage] = []

    # Pool of workers to assign (FIFO — same logic as Node.js orchestrator)
    available_workers = [w for w in workers if w.id not in used_worker_ids]

    newly_used_vehicle_ids: set[str] = set()
    newly_used_worker_ids:  set[str] = set()

    for assignment in assignments:
        if not available_workers:
            # No more workers — remaining packages are unscheduled
            for pkg_idx in assignment.package_indices:
                unscheduled.append(UnscheduledPackage(
                    packageId=packages[pkg_idx].id,
                    reason="No workers available",
                ))
            continue

        # Resolve VehicleInput from the GA's sorted vehicle index
        veh_input = ga_veh_idx_to_input.get(assignment.vehicle_idx, vehicles[0])
        worker      = available_workers[0]
        pkg_inputs  = [packages[i] for i in assignment.package_indices]

        # Validate capacity (GA should have handled this, but double-check)
        total_w = sum(p.weight for p in pkg_inputs)
        total_v = sum(p.volume for p in pkg_inputs)
        has_fragile = any(p.isFragile for p in pkg_inputs)

        cap_ok = (
            total_w <= veh_input.maxWeight * CAPACITY_BUFFER
            and total_v <= veh_input.maxVolume * CAPACITY_BUFFER
            and (not has_fragile or veh_input.supportsFragile)
        )

        if not cap_ok or not pkg_inputs:
            # ── Safety net: GA produced a bad assignment for this group ──────
            # Before giving up, try every remaining vehicle to find one that
            # can actually carry this load.  This rescues packages that the GA
            # incorrectly split or assigned to the wrong vehicle — the most
            # common failure mode being "everything fits in the large truck but
            # the GA spread it across smaller vehicles instead."
            rescue_vehicle = _find_rescue_vehicle(
                vehicles, total_w, total_v, has_fragile,
                used_ids=newly_used_vehicle_ids
            )
            if rescue_vehicle is not None:
                # Swap the GA-assigned vehicle for the rescue vehicle and
                # let the route-building continue normally below.
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

        # ── Build stop points ─────────────────────────────────────────────────
        stops = _build_stops(pkg_inputs, route_type, origin=origin)

        if not stops:
            for p in pkg_inputs:
                unscheduled.append(UnscheduledPackage(
                    packageId=p.id, reason="Could not build stop points"
                ))
            continue

        # ── Slice sub-matrix for this vehicle from the global matrix ──────────
        # Transporter pass: global_dm is None (no OSRM call was made).
        # Deliverer pass:   slice in-memory, no HTTP call, < 1 ms.
        sub_matrix: list[list[float]] | None = None
        stop_index_map: dict[str, int] = {}

        if global_dm is not None and global_dm.source == "osrm":
            # Build an ordered list of global matrix indices for this vehicle:
            # [origin_idx, stop0_idx, stop1_idx, ...]
            veh_global_indices = [global_dm.index_of(origin)]
            for stop in stops:
                idx = global_dm.index_of(stop.coords)
                veh_global_indices.append(idx if idx is not None else 0)

            # Slice: sub_matrix[i][j] = global_dm.matrix[veh_global_indices[i]][veh_global_indices[j]]
            sub_matrix = global_dm.slice(veh_global_indices)

            # Map stop IDs to sub-matrix positions (0 = origin)
            stop_index_map = {"__origin__": 0}
            for si, stop in enumerate(stops):
                stop_index_map[stop.id] = si + 1

        # ── Route optimization (nearest-neighbour + 2-opt) ────────────────────
        route_result = optimised_route(
            origin=origin,
            stops=stops,
            route_type=route_type,
            dist_matrix=sub_matrix,
            stop_index_map=stop_index_map if sub_matrix else None,
        )

        # ── Compute time estimate ─────────────────────────────────────────────
        total_drive = sum(route_result.segment_drive_minutes)
        total_dwell = len(stops) * dwell_minutes
        total_time  = total_drive + total_dwell

        # ── Build output ──────────────────────────────────────────────────────
        stop_outputs = [
            _build_stop_output(stop, packages, route_type)
            for stop in route_result.ordered_stops
        ]
        all_pkg_ids = [p.id for p in pkg_inputs]

        # Inter-branch distance is always 0.0 — branch coordinates are not in
        # the request and are resolved by Node.js at persist time.  Label the
        # source "n/a" so Node.js knows this is a placeholder, not a real
        # distance calculation.
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

        route = RouteOutput(
            vehicleId=veh_input.id,
            workerId=worker.id,
            routeType="inter_branch" if route_type == "inter_branch" else "local_delivery",
            stops=stop_outputs,
            packageIds=all_pkg_ids,
            totalWeight=round(total_w, 2),
            totalVolume=round(total_v, 4),
            distanceKm=distance_km,
            estimatedTimeMinutes=total_time,
            distanceSource=distance_source,
        )

        routes.append(route)

        # Mark vehicle and worker as used
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

def _to_ga_packages(packages: list[PackageInput], route_type: str) -> list[PackageGA]:
    result = []
    for i, p in enumerate(packages):
        if route_type == "local_delivery":
            coords = p.destination.coordinates if p.destination else None
        else:
            # For transporter packages, use None — GA estimates cluster cost
            # differently (branch coordinates are fetched at route-build time)
            coords = None

        result.append(PackageGA(
            idx=i,
            weight=p.weight,
            volume=p.volume,
            is_fragile=p.isFragile,
            coords=coords,
            priority=PRIORITY_MAP.get(p.deliveryPriority, 2),
        ))
    return result


_VEHICLE_TYPE_RANK: dict[str, int] = {
    "motorcycle":  0,
    "car":         1,
    "van":         2,
    "small_truck": 3,
    "large_truck": 4,
}


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


def _build_stops(packages: list[PackageInput], route_type: str, origin: tuple[float, float] | None = None) -> list[StopPoint]:
    """
    Groups packages by delivery location into StopPoints.
    Mirrors the coordinate-grouping logic in delivererRouteBuilder.ts.

    For transporter packages, coordinates are not in the request (only
    destinationBranchId is). We use origin as a placeholder so GA and routing
    can still run. Node.js resolves real branch coordinates at persist time.
    """
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

        else:  # inter_branch
            if not pkg.destinationBranchId:
                continue
            key = pkg.destinationBranchId
            if key not in groups:
                # Use origin as placeholder coords for distance estimation.
                # The real branch coordinates are fetched by Node.js at persist time.
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
    vehicles:     list[VehicleInput],
    total_w:      float,
    total_v:      float,
    has_fragile:  bool,
    used_ids:     set[str],
) -> VehicleInput | None:
    """
    Safety net: finds the smallest available vehicle that can carry a load
    (total_w, total_v) when the GA assigned it to the wrong vehicle.

    Searches all vehicles — including ones not yet used this pass — and
    returns the smallest fitting one to respect the consolidation objective.
    Vehicles already used this pass (used_ids) are excluded because each
    vehicle can only be on one route.
    """
    candidates = [
        v for v in vehicles
        if v.id not in used_ids
        and (not has_fragile or v.supportsFragile)
        and total_w <= v.maxWeight * CAPACITY_BUFFER
        and total_v <= v.maxVolume * CAPACITY_BUFFER
    ]
    if not candidates:
        return None
    # Pick the smallest fitting vehicle (same preference as the GA)
    _rank = {"motorcycle": 0, "car": 1, "van": 2, "small_truck": 3, "large_truck": 4}
    candidates.sort(key=lambda v: (_rank.get(v.type, 2), v.maxWeight))
    return candidates[0]


def _build_stop_output(
    stop: StopPoint,
    packages: list[PackageInput],
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
            # stop.id IS the destinationBranchId for transporter stops
            destinationBranchId=stop.meta.get("destinationBranchId", stop.id),
        )