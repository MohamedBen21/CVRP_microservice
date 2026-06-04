# ─────────────────────────────────────────────────────────────────────────────
#  services/optimizer_pipeline.py
#  Full CVRP / hub-routing pipeline for one branch or hub.
#
#  Architecture (after fix):
#  ──────────────────────────
#  The orchestrator sends workers who ALREADY have a vehicle assigned to them
#  (currentVehicleId set by the manager).  Python does NOT assign vehicles —
#  it only assigns packages or manifests to the fixed worker-vehicle pairs.
#
#  Each worker in the request has:
#    worker.preferredVehicleId = their pre-assigned vehicle's _id
#
#  And the vehicles list contains exactly those pre-assigned vehicles.
#
#  The pipeline builds one route per worker, using their specific vehicle
#  for capacity checks.  The GA still runs to decide WHICH packages go to
#  WHICH worker, but the vehicle for each worker is fixed upfront.
#
#  Fix (Problems 1 & 4):
#    Removed _lock_preferred_pairings().  Instead, _build_worker_vehicle_map()
#    directly maps each worker to their pre-assigned vehicle.  Workers without
#    a matching vehicle in the payload are skipped with a warning.
#
#  Fix (Problem 5):
#    _optimize_pass() returns early when no workers have assigned vehicles,
#    marking all packages unscheduled.  The clustering guard in clustering.py
#    handles n_vehicles=0 defensively.
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

DELIVERER_DWELL  = 8
HUB_BRANCH_DWELL = 20
HUB_HUB_DWELL    = 30

_VEHICLE_TYPE_RANK: dict[str, int] = {
    "motorcycle":  0,
    "car":         1,
    "van":         2,
    "small_truck": 3,
    "large_truck": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER-VEHICLE MAP BUILDER
#  Fix (Problems 1 & 4): simple direct mapping — each worker arrives with
#  their vehicle already assigned.  No GA vehicle selection needed.
# ─────────────────────────────────────────────────────────────────────────────

def _build_worker_vehicle_map(
    workers:  list[WorkerInput],
    vehicles: list[VehicleInput],
) -> dict[str, VehicleInput]:
    """
    Maps worker_id → VehicleInput using worker.preferredVehicleId.

    Every worker in the CVRP request should have preferredVehicleId set
    (populated by the orchestrator from worker.currentVehicleId in the DB).
    Workers without a matching vehicle in the payload are excluded from
    routing — they will not receive any packages/manifests this run.
    """
    vehicle_by_id = {v.id: v for v in vehicles}
    result: dict[str, VehicleInput] = {}

    for w in workers:
        pref = getattr(w, "preferredVehicleId", None)
        if pref and pref in vehicle_by_id:
            result[w.id] = vehicle_by_id[pref]
        else:
            logger.warning(
                f"[pipeline] Worker {w.id} has no matching vehicle in payload "
                f"(preferredVehicleId={pref}) — skipping this worker"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(req: OptimizeRequest) -> OptimizeResponse:
    origin = req.branch.coordinates

    hub_to_hub_workers    = [w for w in req.workers if w.role == "transporter" and w.transporterType == "hub_to_hub"]
    hub_to_branch_workers = [w for w in req.workers if w.role == "transporter" and w.transporterType == "hub_to_branch"]
    legacy_transporters   = [w for w in req.workers if w.role == "transporter" and not w.transporterType]
    deliverers            = [w for w in req.workers if w.role == "deliverer"]

    h2h_hub_ids: set[str] = set()
    for w in hub_to_hub_workers:
        for hub_id in (w.assignedLine or []):
            h2h_hub_ids.add(hub_id)

    h2h_manifests = [m for m in req.manifests if m.destinationBranchId in h2h_hub_ids]
    h2b_manifests = [m for m in req.manifests if m.destinationBranchId not in h2h_hub_ids]

    deliverer_pkgs = [
        p for p in req.packages
        if p.deliveryType == "home" and p.destination and p.destination.coordinates
    ]
    legacy_transporter_pkgs = [
        p for p in req.packages
        if p.destinationBranchId and p not in deliverer_pkgs
    ]
    unroutable_packages = [
        p for p in req.packages
        if p not in deliverer_pkgs and p not in legacy_transporter_pkgs
    ]

    all_routes:                list[RouteOutput]          = []
    all_unscheduled:           list[UnscheduledPackage]   = []
    all_unscheduled_manifests: list[UnscheduledManifest]  = []

    used_worker_ids: set[str] = set()

    for p in unroutable_packages:
        all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason="Missing destination coordinates"))

    # ── Pass 1: hub_to_hub ────────────────────────────────────────────────────
    if h2h_manifests and hub_to_hub_workers:
        routes, unsched, used_w = _optimize_hub_to_hub(
            manifests=h2h_manifests,
            vehicles=req.vehicles,
            workers=hub_to_hub_workers,
            origin=origin,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(routes)
        all_unscheduled_manifests.extend(unsched)
        used_worker_ids.update(used_w)
    elif h2h_manifests:
        for m in h2h_manifests:
            all_unscheduled_manifests.append(UnscheduledManifest(manifestId=m.id, reason="No hub_to_hub workers available"))

    # ── Pass 2: hub_to_branch ─────────────────────────────────────────────────
    if h2b_manifests and hub_to_branch_workers:
        routes, unsched, used_w = _optimize_hub_to_branch(
            manifests=h2b_manifests,
            vehicles=req.vehicles,
            workers=hub_to_branch_workers,
            origin=origin,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(routes)
        all_unscheduled_manifests.extend(unsched)
        used_worker_ids.update(used_w)
    elif h2b_manifests:
        for m in h2b_manifests:
            all_unscheduled_manifests.append(UnscheduledManifest(manifestId=m.id, reason="No hub_to_branch workers available"))

    # ── Pass 3: legacy transporter ────────────────────────────────────────────
    if legacy_transporter_pkgs and legacy_transporters:
        routes, unsched, used_w = _optimize_pass(
            packages=legacy_transporter_pkgs,
            vehicles=req.vehicles,
            workers=legacy_transporters,
            origin=origin,
            route_type="inter_branch",
            dwell_minutes=HUB_BRANCH_DWELL,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(routes)
        all_unscheduled.extend(unsched)
        used_worker_ids.update(used_w)
    elif legacy_transporter_pkgs:
        for p in legacy_transporter_pkgs:
            all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason="No legacy transporters available"))

    # ── Pass 4: deliverers ────────────────────────────────────────────────────
    if deliverer_pkgs and deliverers:
        routes, unsched, used_w = _optimize_pass(
            packages=deliverer_pkgs,
            vehicles=req.vehicles,
            workers=deliverers,
            origin=origin,
            route_type="local_delivery",
            dwell_minutes=DELIVERER_DWELL,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(routes)
        all_unscheduled.extend(unsched)
        used_worker_ids.update(used_w)
    elif deliverer_pkgs:
        for p in deliverer_pkgs:
            all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason="No deliverers available"))

    return OptimizeResponse(
        routes=all_routes,
        unscheduled=all_unscheduled,
        unscheduledManifests=all_unscheduled_manifests,
        meta={
            "totalPackages":        len(req.packages),
            "totalManifests":       len(req.manifests),
            "scheduledPackages":    sum(len(r.packageIds) for r in all_routes),
            "scheduledManifests":   sum(len(r.manifestIds) for r in all_routes),
            "unscheduledPackages":  len(all_unscheduled),
            "unscheduledManifests": len(all_unscheduled_manifests),
            "routesCreated":        len(all_routes),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 1: HUB-TO-HUB
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_hub_to_hub(
    manifests:       list[ManifestInput],
    vehicles:        list[VehicleInput],
    workers:         list[WorkerInput],
    origin:          tuple[float, float],
    used_worker_ids: set[str],
) -> tuple[list[RouteOutput], list[UnscheduledManifest], set[str]]:

    routes:      list[RouteOutput]         = []
    unscheduled: list[UnscheduledManifest] = []
    newly_used:  set[str]                  = set()

    # Build worker→vehicle map from pre-assigned pairings
    worker_vehicle = _build_worker_vehicle_map(workers, vehicles)

    available_workers = [
        w for w in workers
        if w.id not in used_worker_ids and w.id in worker_vehicle
    ]

    if not available_workers:
        for m in manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id, reason="No hub_to_hub workers with vehicles available"
            ))
        return routes, unscheduled, newly_used

    by_leg: dict[tuple[str, str], list[ManifestInput]] = {}
    for m in manifests:
        key = (m.originBranchId, m.destinationBranchId)
        by_leg.setdefault(key, []).append(m)

    def is_outbound(leg: tuple[str, str]) -> bool:
        sample = by_leg[leg][0] if by_leg[leg] else None
        if not sample:
            return False
        return (
            abs(sample.originCoordinates[0] - origin[0]) < 0.001
            and abs(sample.originCoordinates[1] - origin[1]) < 0.001
        )

    outbound_legs = [leg for leg in by_leg if is_outbound(leg)]
    return_legs   = [leg for leg in by_leg if not is_outbound(leg)]

    def _process_leg(origin_id: str, dest_id: str, leg_manifests: list[ManifestInput]) -> None:
        if not leg_manifests:
            return

        eligible = [
            w for w in available_workers
            if w.assignedLine
            and origin_id in w.assignedLine
            and dest_id   in w.assignedLine
            and w.id not in newly_used
        ]
        if not eligible:
            for m in leg_manifests:
                unscheduled.append(UnscheduledManifest(
                    manifestId=m.id,
                    reason=f"No hub_to_hub worker for leg {origin_id} → {dest_id}",
                ))
            return

        sorted_manifests = sorted(leg_manifests, key=lambda m: -m.totalWeight)

        for worker in eligible:
            if not sorted_manifests:
                break

            veh = worker_vehicle[worker.id]
            batch:    list[ManifestInput] = []
            leftover: list[ManifestInput] = []
            running_w = 0.0
            running_v = 0.0

            for m in sorted_manifests:
                if (
                    running_w + m.totalWeight <= veh.maxWeight * CAPACITY_BUFFER
                    and running_v + m.totalVolume <= veh.maxVolume * CAPACITY_BUFFER
                ):
                    batch.append(m)
                    running_w += m.totalWeight
                    running_v += m.totalVolume
                else:
                    leftover.append(m)

            sorted_manifests = leftover

            if not batch:
                continue

            dest_coords = batch[0].destinationCoordinates
            stop = StopOutput(
                coordinates=dest_coords,
                manifestIds=[m.id for m in batch],
                destinationBranchId=dest_id,
            )
            route = RouteOutput(
                vehicleId=veh.id,
                workerId=worker.id,
                routeType="hub_to_hub",
                stops=[stop],
                packageIds=[],
                manifestIds=[m.id for m in batch],
                totalWeight=round(running_w, 2),
                totalVolume=round(running_v, 4),
                distanceKm=0.0,
                estimatedTimeMinutes=HUB_HUB_DWELL,
                distanceSource="n/a",
                originBranchId=origin_id,
            )
            routes.append(route)
            newly_used.add(worker.id)

        for m in sorted_manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason=f"Capacity exceeded on all workers for leg {origin_id} → {dest_id}",
            ))

    for (orig, dest) in outbound_legs:
        _process_leg(orig, dest, by_leg[(orig, dest)])
        reverse_key = (dest, orig)
        if reverse_key in by_leg:
            _process_leg(dest, orig, by_leg.pop(reverse_key))

    for (orig, dest) in return_legs:
        if (orig, dest) in by_leg:
            _process_leg(orig, dest, by_leg[(orig, dest)])

    return routes, unscheduled, newly_used


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 2: HUB-TO-BRANCH
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_hub_to_branch(
    manifests:       list[ManifestInput],
    vehicles:        list[VehicleInput],
    workers:         list[WorkerInput],
    origin:          tuple[float, float],
    used_worker_ids: set[str],
) -> tuple[list[RouteOutput], list[UnscheduledManifest], set[str]]:

    routes:      list[RouteOutput]         = []
    unscheduled: list[UnscheduledManifest] = []
    newly_used:  set[str]                  = set()

    worker_vehicle = _build_worker_vehicle_map(workers, vehicles)

    available_workers = [
        w for w in workers
        if w.id not in used_worker_ids and w.id in worker_vehicle
    ]

    if not available_workers:
        for m in manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id, reason="No hub_to_branch workers with vehicles available"
            ))
        return routes, unscheduled, newly_used

    # Pre-filter: skip manifests with no worker covering their destination
    branch_to_workers: dict[str, list[WorkerInput]] = {}
    for w in available_workers:
        for branch_id in (w.assignedBranches or []):
            branch_to_workers.setdefault(branch_id, []).append(w)

    covered:   list[ManifestInput] = []
    for m in manifests:
        if m.destinationBranchId in branch_to_workers:
            covered.append(m)
        else:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason=f"No hub_to_branch worker covers branch {m.destinationBranchId}",
            ))

    if not covered:
        return routes, unscheduled, newly_used

    # Assign manifests to workers — greedy by weight, prefer least-loaded worker
    worker_manifests: dict[str, list[ManifestInput]] = {w.id: [] for w in available_workers}
    worker_w_used:    dict[str, float]                = {w.id: 0.0 for w in available_workers}
    worker_v_used:    dict[str, float]                = {w.id: 0.0 for w in available_workers}

    for m in sorted(covered, key=lambda m: -m.totalWeight):
        dest = m.destinationBranchId
        candidates = [
            w for w in available_workers
            if dest in (w.assignedBranches or [])
        ]
        assigned = False
        for w in sorted(candidates, key=lambda w: worker_w_used[w.id]):
            veh = worker_vehicle[w.id]
            new_w = worker_w_used[w.id] + m.totalWeight
            new_v = worker_v_used[w.id] + m.totalVolume
            if new_w <= veh.maxWeight * CAPACITY_BUFFER and new_v <= veh.maxVolume * CAPACITY_BUFFER:
                worker_manifests[w.id].append(m)
                worker_w_used[w.id] = new_w
                worker_v_used[w.id] = new_v
                assigned = True
                break
        if not assigned:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id, reason="Capacity exceeded on all workers covering this branch"
            ))

    # Build a route for each worker that has manifests
    for worker in available_workers:
        wm = worker_manifests[worker.id]
        if not wm:
            continue

        veh   = worker_vehicle[worker.id]
        stops = _build_manifest_stops(wm)

        all_stop_coords = [s.coords for s in stops]
        global_dm = GlobalDistanceMatrix.build(origin, all_stop_coords)

        sub_matrix:     list[list[float]] | None = None
        stop_index_map: dict[str, int]           = {}
        if global_dm.source == "osrm":
            indices = [global_dm.index_of(origin)]
            for s in stops:
                idx = global_dm.index_of(s.coords)
                indices.append(idx if idx is not None else 0)
            sub_matrix = global_dm.slice(indices)
            stop_index_map = {"__origin__": 0}
            for si, s in enumerate(stops):
                stop_index_map[s.id] = si + 1

        route_result = optimised_route(
            origin=origin,
            stops=stops,
            route_type="inter_branch",
            dist_matrix=sub_matrix,
            stop_index_map=stop_index_map if sub_matrix else None,
        )

        total_time = sum(route_result.segment_drive_minutes) + len(stops) * HUB_BRANCH_DWELL

        manifest_by_dest: dict[str, list[ManifestInput]] = {}
        for m in wm:
            manifest_by_dest.setdefault(m.destinationBranchId, []).append(m)

        stop_outputs = [
            StopOutput(
                coordinates=sp.coords,
                manifestIds=[m.id for m in manifest_by_dest.get(sp.meta.get("destinationBranchId", sp.id), [])],
                destinationBranchId=sp.meta.get("destinationBranchId", sp.id),
            )
            for sp in route_result.ordered_stops
        ]

        routes.append(RouteOutput(
            vehicleId=veh.id,
            workerId=worker.id,
            routeType="hub_to_branch",
            stops=stop_outputs,
            packageIds=[],
            manifestIds=[m.id for m in wm],
            totalWeight=round(worker_w_used[worker.id], 2),
            totalVolume=round(worker_v_used[worker.id], 4),
            distanceKm=round(route_result.total_distance_km, 2),
            estimatedTimeMinutes=total_time,
            distanceSource=route_result.distance_source,
        ))
        newly_used.add(worker.id)

    return routes, unscheduled, newly_used


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 3/4: PACKAGES (legacy transporter + deliverer)
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_pass(
    packages:        list[PackageInput],
    vehicles:        list[VehicleInput],
    workers:         list[WorkerInput],
    origin:          tuple[float, float],
    route_type:      str,
    dwell_minutes:   int,
    used_worker_ids: set[str],
) -> tuple[list[RouteOutput], list[UnscheduledPackage], set[str]]:

    routes:      list[RouteOutput]        = []
    unscheduled: list[UnscheduledPackage] = []
    newly_used:  set[str]                 = set()

    # Build worker→vehicle map from pre-assigned pairings
    worker_vehicle = _build_worker_vehicle_map(workers, vehicles)

    available_workers = [
        w for w in workers
        if w.id not in used_worker_ids and w.id in worker_vehicle
    ]

    # Fix (Problem 5): no workers with vehicles → skip clustering entirely
    if not available_workers:
        logger.warning(
            f"[pipeline] [{route_type}] No workers with vehicles — "
            f"marking all {len(packages)} packages unscheduled"
        )
        return (
            [],
            [UnscheduledPackage(packageId=p.id, reason="No workers with assigned vehicles available")
             for p in packages],
            newly_used,
        )

    # Build GA inputs — vehicles list is derived from available workers' vehicles
    # so the GA knows which capacity constraints apply to each "vehicle slot"
    worker_vehicles_ordered = [worker_vehicle[w.id] for w in available_workers]

    ga_packages = _to_ga_packages(packages, route_type)
    ga_vehicles = _to_ga_vehicles(worker_vehicles_ordered)

    # Cluster packages
    if route_type == "local_delivery":
        coords = [
            p.destination.coordinates if p.destination else origin
            for p in packages
        ]
        # Fix (Problem 5): clustering guard — n_vehicles = len(available_workers)
        # which is guaranteed > 0 at this point (checked above)
        _clusters = cluster_deliverer_packages(coords, len(available_workers))
    else:
        branch_ids = [p.destinationBranchId for p in packages]
        _clusters  = cluster_transporter_packages(branch_ids)

    # Build global distance matrix (one OSRM call for all stops)
    if route_type == "local_delivery":
        all_stop_coords: list[tuple[float, float]] = [
            p.destination.coordinates if p.destination else origin
            for p in packages
        ]
        global_dm = GlobalDistanceMatrix.build(origin, all_stop_coords)
        pkg_to_gidx: dict[int, int] = {}
        for i, p in enumerate(packages):
            c = p.destination.coordinates if p.destination else origin
            idx = global_dm.index_of(c)
            if idx is not None:
                pkg_to_gidx[i] = idx
        n = len(packages)
        ga_dist_matrix: list[list[float]] | None = (
            [[global_dm.matrix[pkg_to_gidx.get(i, 0)][pkg_to_gidx.get(j, 0)] for j in range(n)] for i in range(n)]
            if global_dm.source == "osrm" else None
        )
    else:
        global_dm      = None
        ga_dist_matrix = None

    # Run GA — assigns package indices to vehicle slots (0..n_workers-1)
    is_deliverer = (route_type == "local_delivery")
    assignments, sorted_ga_vehicles = run_genetic_assignment(
        packages=ga_packages,
        vehicles=ga_vehicles,
        origin_coords=origin,
        is_deliverer=is_deliverer,
        dist_matrix=ga_dist_matrix,
    )

    # sorted_ga_vehicles[i].idx is the index into worker_vehicles_ordered
    # → available_workers[idx] is the corresponding worker
    ga_veh_idx_to_worker_and_vehicle: dict[int, tuple[WorkerInput, VehicleInput]] = {}
    for i, sv in enumerate(sorted_ga_vehicles):
        worker = available_workers[sv.idx]
        veh    = worker_vehicles_ordered[sv.idx]
        ga_veh_idx_to_worker_and_vehicle[i] = (worker, veh)

    for assignment in assignments:
        if assignment.vehicle_idx not in ga_veh_idx_to_worker_and_vehicle:
            for pkg_idx in assignment.package_indices:
                unscheduled.append(UnscheduledPackage(
                    packageId=packages[pkg_idx].id, reason="GA returned invalid vehicle index"
                ))
            continue

        worker, veh = ga_veh_idx_to_worker_and_vehicle[assignment.vehicle_idx]

        if worker.id in newly_used:
            # Worker already assigned (shouldn't happen with correct GA vehicle count)
            for pkg_idx in assignment.package_indices:
                unscheduled.append(UnscheduledPackage(
                    packageId=packages[pkg_idx].id, reason="Worker already assigned to another route"
                ))
            continue

        pkg_inputs  = [packages[i] for i in assignment.package_indices]
        total_w     = sum(p.weight for p in pkg_inputs)
        total_v     = sum(p.volume for p in pkg_inputs)
        has_fragile = any(p.isFragile for p in pkg_inputs)

        # Capacity check against the worker's actual vehicle
        cap_ok = (
            total_w <= veh.maxWeight * CAPACITY_BUFFER
            and total_v <= veh.maxVolume * CAPACITY_BUFFER
            and (not has_fragile or veh.supportsFragile)
        )

        if not cap_ok or not pkg_inputs:
            for p in pkg_inputs:
                if has_fragile and not veh.supportsFragile:
                    reason = f"Vehicle {veh.registrationNumber} does not support fragile packages"
                elif total_w > veh.maxWeight * CAPACITY_BUFFER:
                    reason = f"Exceeds vehicle weight capacity ({total_w:.1f}kg > {veh.maxWeight * CAPACITY_BUFFER:.1f}kg)"
                else:
                    reason = f"Exceeds vehicle volume capacity"
                unscheduled.append(UnscheduledPackage(packageId=p.id, reason=reason))
            continue

        stops = _build_stops(pkg_inputs, route_type, origin=origin)
        if not stops:
            for p in pkg_inputs:
                unscheduled.append(UnscheduledPackage(packageId=p.id, reason="Could not build stop points"))
            continue

        sub_matrix:     list[list[float]] | None = None
        stop_index_map: dict[str, int]           = {}
        if global_dm is not None and global_dm.source == "osrm":
            indices = [global_dm.index_of(origin)]
            for s in stops:
                idx = global_dm.index_of(s.coords)
                indices.append(idx if idx is not None else 0)
            sub_matrix = global_dm.slice(indices)
            stop_index_map = {"__origin__": 0}
            for si, s in enumerate(stops):
                stop_index_map[s.id] = si + 1

        route_result = optimised_route(
            origin=origin,
            stops=stops,
            route_type=route_type,
            dist_matrix=sub_matrix,
            stop_index_map=stop_index_map if sub_matrix else None,
        )

        total_time = sum(route_result.segment_drive_minutes) + len(stops) * dwell_minutes
        stop_outputs = [_build_stop_output(s, packages, route_type) for s in route_result.ordered_stops]

        output_route_type = "hub_to_branch" if route_type == "inter_branch" else "local_delivery"
        distance_km       = 0.0 if route_type == "inter_branch" else round(route_result.total_distance_km, 2)
        distance_source   = "n/a" if route_type == "inter_branch" else route_result.distance_source

        routes.append(RouteOutput(
            vehicleId=veh.id,
            workerId=worker.id,
            routeType=output_route_type,
            stops=stop_outputs,
            packageIds=[p.id for p in pkg_inputs],
            manifestIds=[],
            totalWeight=round(total_w, 2),
            totalVolume=round(total_v, 4),
            distanceKm=distance_km,
            estimatedTimeMinutes=total_time,
            distanceSource=distance_source,
        ))
        newly_used.add(worker.id)

        logger.debug(
            f"[pipeline] [{route_type}] vehicle={veh.registrationNumber} "
            f"worker={worker.id} pkg={len(pkg_inputs)} "
            f"dist={route_result.total_distance_km:.1f}km time={total_time}min"
        )

    return routes, unscheduled, newly_used


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_manifest_stops(manifests: list[ManifestInput]) -> list[StopPoint]:
    groups: dict[str, dict] = {}
    for m in manifests:
        key = m.destinationBranchId
        if key not in groups:
            groups[key] = {"coords": m.destinationCoordinates, "ids": [], "meta": {"destinationBranchId": m.destinationBranchId}}
        groups[key]["ids"].append(m.id)
    return [StopPoint(stop_id=k, coords=g["coords"], package_ids=g["ids"], meta=g["meta"]) for k, g in groups.items()]


def _to_ga_packages(packages: list[PackageInput], route_type: str) -> list[PackageGA]:
    return [
        PackageGA(
            idx=i,
            weight=p.weight,
            volume=p.volume,
            is_fragile=p.isFragile,
            coords=p.destination.coordinates if (route_type == "local_delivery" and p.destination) else None,
            priority=PRIORITY_MAP.get(p.deliveryPriority, 2),
        )
        for i, p in enumerate(packages)
    ]


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
    groups: dict[str, dict] = {}
    for pkg in packages:
        if route_type == "local_delivery":
            if not pkg.destination or not pkg.destination.coordinates:
                continue
            lng, lat = pkg.destination.coordinates
            key = f"{lng:.5f},{lat:.5f}"
            if key not in groups:
                groups[key] = {"coords": (lng, lat), "pkg_ids": [], "meta": {"address": pkg.destination.address, "recipientName": pkg.destination.recipientName}}
            groups[key]["pkg_ids"].append(pkg.id)
        else:
            if not pkg.destinationBranchId:
                continue
            key = pkg.destinationBranchId
            if key not in groups:
                groups[key] = {"coords": origin or (0.0, 0.0), "pkg_ids": [], "meta": {"destinationBranchId": pkg.destinationBranchId}}
            groups[key]["pkg_ids"].append(pkg.id)
    return [StopPoint(stop_id=k, coords=g["coords"], package_ids=g["pkg_ids"], meta=g["meta"]) for k, g in groups.items()]


def _build_stop_output(stop: StopPoint, packages: list[PackageInput], route_type: str) -> StopOutput:
    if route_type == "local_delivery":
        return StopOutput(coordinates=stop.coords, packageIds=stop.package_ids, address=stop.meta.get("address", ""), recipientName=stop.meta.get("recipientName", ""))
    else:
        return StopOutput(coordinates=stop.coords, packageIds=stop.package_ids, destinationBranchId=stop.meta.get("destinationBranchId", stop.id))