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

def run_optimization(req: OptimizeRequest) -> OptimizeResponse:
    origin = req.branch.coordinates  # [lng, lat]

    # ── Segment workers by role + sub-type ────────────────────────────────────
    hub_to_hub_workers    = [
        w for w in req.workers
        if w.role == "transporter" and w.transporterType == "hub_to_hub"
    ]
    hub_to_branch_workers = [
        w for w in req.workers
        if w.role == "transporter" and w.transporterType == "hub_to_branch"
    ]
    legacy_transporters   = [
        w for w in req.workers
        if w.role == "transporter" and not w.transporterType
    ]
    deliverers            = [w for w in req.workers if w.role == "deliverer"]

    # ── Segment manifests by worker type ────────────────────────────────────
    # Collect every hub ID that any hub_to_hub worker services.
    # Manifests whose destinationBranchId is one of these IDs belong to the
    # hub_to_hub pass.  All other manifests belong to the hub_to_branch pass.
    # This prevents the same manifest from being processed by both passes and
    # avoids inflated unscheduled counts.
    h2h_hub_ids: set[str] = set()
    for w in hub_to_hub_workers:
        for hub_id in (w.assignedLine or []):
            h2h_hub_ids.add(hub_id)

    h2h_manifests = [m for m in req.manifests if m.destinationBranchId in h2h_hub_ids]
    h2b_manifests = [m for m in req.manifests if m.destinationBranchId not in h2h_hub_ids]

    # Packages that have no valid destination info
    unroutable_packages = [
        p for p in req.packages
        if not p.destinationBranchId
        and not (p.deliveryType == "home" and p.destination and p.destination.coordinates)
    ]
    deliverer_pkgs = [
        p for p in req.packages
        if p.deliveryType == "home"
        and p.destination is not None
        and p.destination.coordinates is not None
    ]
    legacy_transporter_pkgs = [
        p for p in req.packages
        if p.destinationBranchId
        and p not in deliverer_pkgs
    ]

    all_routes:               list[RouteOutput]        = []
    all_unscheduled:          list[UnscheduledPackage]  = []
    all_unscheduled_manifests: list[UnscheduledManifest] = []

    used_vehicle_ids: set[str] = set()
    used_worker_ids:  set[str] = set()

    # Mark unroutable packages up front
    for p in unroutable_packages:
        all_unscheduled.append(
            UnscheduledPackage(packageId=p.id, reason="Missing destination coordinates")
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  PASS 1 — HUB-TO-HUB  (only h2h_manifests — hub-destined bags)
    # ─────────────────────────────────────────────────────────────────────────
    if h2h_manifests and hub_to_hub_workers:
        available_vehicles = [v for v in req.vehicles if v.id not in used_vehicle_ids]
        h2h_routes, h2h_unscheduled, h2h_used_v, h2h_used_w = _optimize_hub_to_hub(
            manifests=h2h_manifests,
            vehicles=available_vehicles,
            workers=hub_to_hub_workers,
            origin=origin,
            used_vehicle_ids=used_vehicle_ids,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(h2h_routes)
        all_unscheduled_manifests.extend(h2h_unscheduled)
        used_vehicle_ids.update(h2h_used_v)
        used_worker_ids.update(h2h_used_w)
    elif h2h_manifests and not hub_to_hub_workers:
        for m in h2h_manifests:
            all_unscheduled_manifests.append(
                UnscheduledManifest(manifestId=m.id, reason="No hub_to_hub workers available for hub-destined manifests")
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  PASS 2 — HUB-TO-BRANCH (only h2b_manifests → local branches)
    # ─────────────────────────────────────────────────────────────────────────
    if h2b_manifests and hub_to_branch_workers:
        available_vehicles = [v for v in req.vehicles if v.id not in used_vehicle_ids]
        h2b_routes, h2b_unscheduled, h2b_used_v, h2b_used_w = _optimize_hub_to_branch(
            manifests=h2b_manifests,
            vehicles=available_vehicles,
            workers=hub_to_branch_workers,
            origin=origin,
            used_vehicle_ids=used_vehicle_ids,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(h2b_routes)
        all_unscheduled_manifests.extend(h2b_unscheduled)
        used_vehicle_ids.update(h2b_used_v)
        used_worker_ids.update(h2b_used_w)
    elif h2b_manifests and not hub_to_branch_workers:
        for m in h2b_manifests:
            all_unscheduled_manifests.append(
                UnscheduledManifest(manifestId=m.id, reason="No hub_to_branch workers available for branch-destined manifests")
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  PASS 3 — LEGACY TRANSPORTER (raw packages, inter_branch)
    #  Retained for backward compatibility with branches not yet on hub model.
    # ─────────────────────────────────────────────────────────────────────────
    if legacy_transporter_pkgs and legacy_transporters:
        available_vehicles = [v for v in req.vehicles if v.id not in used_vehicle_ids]
        t_routes, t_unscheduled, t_used_v, t_used_w = _optimize_pass(
            packages=legacy_transporter_pkgs,
            vehicles=available_vehicles,
            workers=legacy_transporters,
            origin=origin,
            route_type="inter_branch",
            dwell_minutes=HUB_BRANCH_DWELL,
            used_vehicle_ids=used_vehicle_ids,
            used_worker_ids=used_worker_ids,
        )
        all_routes.extend(t_routes)
        all_unscheduled.extend(t_unscheduled)
        used_vehicle_ids.update(t_used_v)
        used_worker_ids.update(t_used_w)
    elif legacy_transporter_pkgs:
        for p in legacy_transporter_pkgs:
            all_unscheduled.append(
                UnscheduledPackage(packageId=p.id, reason="No legacy transporters available")
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  PASS 4 — DELIVERER (packages → customer addresses)
    # ─────────────────────────────────────────────────────────────────────────
    if deliverer_pkgs and deliverers:
        available_vehicles = [v for v in req.vehicles if v.id not in used_vehicle_ids]
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
    elif deliverer_pkgs:
        for p in deliverer_pkgs:
            reason = (
                "No deliverers available" if not deliverers
                else "No vehicles available"
            )
            all_unscheduled.append(UnscheduledPackage(packageId=p.id, reason=reason))

    return OptimizeResponse(
        routes=all_routes,
        unscheduled=all_unscheduled,
        unscheduledManifests=all_unscheduled_manifests,
        meta={
            "totalPackages":         len(req.packages),
            "totalManifests":        len(req.manifests),
            "h2hManifests":          len(h2h_manifests),
            "h2bManifests":          len(h2b_manifests),
            "scheduledPackages":     sum(len(r.packageIds) for r in all_routes),
            "scheduledManifests":    sum(len(r.manifestIds) for r in all_routes),
            "unscheduledPackages":   len(all_unscheduled),
            "unscheduledManifests":  len(all_unscheduled_manifests),
            "routesCreated":         len(all_routes),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 1 IMPLEMENTATION: HUB-TO-HUB
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_hub_to_hub(
    manifests:        list[ManifestInput],
    vehicles:         list[VehicleInput],
    workers:          list[WorkerInput],   # all transporterType == "hub_to_hub"
    origin:           tuple[float, float],
    used_vehicle_ids: set[str],
    used_worker_ids:  set[str],
) -> tuple[list[RouteOutput], list[UnscheduledManifest], set[str], set[str]]:
    """
    Hub-to-hub routing — same-night round-trip support.

    The manifest list contains BOTH outbound and return manifests:
      • Outbound: originBranchId == planning hub (hub A),  destination == hub B
      • Return:   originBranchId == partner hub  (hub B),  destination == hub A

    Strategy
    ────────
    1. Group manifests by (originBranchId, destinationBranchId) — each unique
       pair is one leg.
    2. For each leg, find workers whose assignedLine covers that origin→destination
       pair and who are NOT yet used.
    3. Pair outbound + return legs to the SAME worker when possible so T1 gets
       both legs in one planning run:
         - Assign T1 to the outbound leg  (hub A → hub B).
         - Reserve T1 for the return leg  (hub B → hub A).
       This means T1 arrives at hub B, drops outbound manifests, picks up the
       pre-built return route immediately — no waiting for the next nightly run.
    4. If there are no return manifests, the outbound route is still created
       normally and T1 stays at hub B until return manifests accumulate.
    5. If return manifests exist but there is no worker to cover them (e.g. a
       one-way line), they are marked unscheduled.

    Vehicle assignment
    ──────────────────
    Each leg gets its own vehicle entry in the route output so Node.js can
    persist two separate RouteModel documents (one per leg).  In practice T1
    drives the same physical truck both ways, but the route documents are
    independent for clean status tracking.
    """
    routes:      list[RouteOutput]         = []
    unscheduled: list[UnscheduledManifest] = []

    newly_used_vehicle_ids: set[str] = set()
    newly_used_worker_ids:  set[str] = set()

    available_workers  = [w for w in workers if w.id not in used_worker_ids]
    available_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]

    if not available_workers or not available_vehicles:
        for m in manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason="No hub_to_hub workers or vehicles available",
            ))
        return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids

    # ── Step 1: group manifests by (origin, destination) leg ─────────────────
    # key = (originBranchId, destinationBranchId)
    by_leg: dict[tuple[str, str], list[ManifestInput]] = {}
    for m in manifests:
        key = (m.originBranchId, m.destinationBranchId)
        by_leg.setdefault(key, []).append(m)

    # ── Step 2: identify line pairs so we can do same-night pairing ───────────
    # A "line pair" is two legs that are the reverse of each other:
    #   leg (A→B) and leg (B→A) belong to the same line.
    # We process legs in outbound-first order (the leg whose origin == the
    # planning hub origin coordinate comes first) so the outbound worker is
    # chosen first and then reserved for the return.

    # Sort legs: outbound first (origin matches the planning-hub origin coords)
    def is_outbound(leg: tuple[str, str]) -> bool:
        """True if this leg departs from the current planning hub."""
        leg_manifests = by_leg[leg]
        if not leg_manifests:
            return False
        # Outbound manifests have originCoordinates ≈ the planning hub origin
        sample = leg_manifests[0]
        return (
            abs(sample.originCoordinates[0] - origin[0]) < 0.001
            and abs(sample.originCoordinates[1] - origin[1]) < 0.001
        )

    outbound_legs = [leg for leg in by_leg if is_outbound(leg)]
    return_legs   = [leg for leg in by_leg if not is_outbound(leg)]

    # ── Step 3: process outbound legs first, then return legs ────────────────
    # For each outbound leg we try to reserve the same worker for the return.
    reserved_for_return: dict[str, str] = {}  # worker_id → return_leg key (str)

    def _process_leg(
        origin_id:  str,
        dest_id:    str,
        leg_manifests: list[ManifestInput],
        preferred_worker_id: str | None = None,
    ) -> None:
        """Assigns manifests for one leg to workers+vehicles, creating RouteOutput(s)."""
        nonlocal routes, unscheduled

        if not leg_manifests:
            return

        # Workers eligible for this leg: assignedLine must contain BOTH hubs
        eligible = [
            w for w in available_workers
            if w.assignedLine
            and origin_id in w.assignedLine
            and dest_id   in w.assignedLine
            and w.id not in newly_used_worker_ids
        ]
        if not eligible:
            for m in leg_manifests:
                unscheduled.append(UnscheduledManifest(
                    manifestId=m.id,
                    reason=f"No hub_to_hub worker for leg {origin_id} → {dest_id}",
                ))
            return

        # If a preferred worker was reserved for this return leg, put them first
        if preferred_worker_id:
            eligible.sort(key=lambda w: 0 if w.id == preferred_worker_id else 1)

        sorted_manifests = sorted(leg_manifests, key=lambda m: -m.totalWeight)

        for worker in eligible:
            if not sorted_manifests:
                break

            veh = _pick_vehicle(available_vehicles, sorted_manifests, newly_used_vehicle_ids)
            if veh is None:
                for m in sorted_manifests:
                    unscheduled.append(UnscheduledManifest(
                        manifestId=m.id,
                        reason="No vehicle with sufficient capacity for hub-to-hub leg",
                    ))
                sorted_manifests = []
                break

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

            dest_coords   = batch[0].destinationCoordinates
            origin_coords = batch[0].originCoordinates

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
                # Tell Node.js which hub this leg departs from so it can set
                # originBranchId correctly on the persisted RouteModel document.
                originBranchId=origin_id,
            )
            routes.append(route)

            newly_used_vehicle_ids.add(veh.id)
            newly_used_worker_ids.add(worker.id)

        for m in sorted_manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason=f"Not enough workers for leg {origin_id} → {dest_id}",
            ))

    # Process outbound legs and reserve workers for their paired return leg
    for (orig, dest) in outbound_legs:
        _process_leg(orig, dest, by_leg[(orig, dest)])

        # Find which worker was just assigned to this outbound leg
        # (the last worker added to newly_used_worker_ids for this leg)
        reverse_key = (dest, orig)
        if reverse_key in by_leg:
            # The worker assigned to the outbound is now in newly_used_worker_ids.
            # We want them to ALSO handle the return.  Remove them temporarily
            # from the used set so _process_leg can pick them for the return.
            # We identify them as the worker whose route was just appended.
            if routes:
                last_worker_id = routes[-1].workerId
                # Temporarily free this worker for the return leg only
                newly_used_worker_ids.discard(last_worker_id)
                _process_leg(dest, orig, by_leg[reverse_key],
                             preferred_worker_id=last_worker_id)
                # Re-mark as used after both legs are assigned
                newly_used_worker_ids.add(last_worker_id)
                # Remove the return leg so it is not processed again below
                del by_leg[reverse_key]

    # Process any remaining return-only legs (no paired outbound this run)
    for (orig, dest) in return_legs:
        if (orig, dest) in by_leg:   # may have been deleted above
            _process_leg(orig, dest, by_leg[(orig, dest)])

    return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 2 IMPLEMENTATION: HUB-TO-BRANCH
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_hub_to_branch(
    manifests:        list[ManifestInput],
    vehicles:         list[VehicleInput],
    workers:          list[WorkerInput],   # all transporterType == "hub_to_branch"
    origin:           tuple[float, float],
    used_vehicle_ids: set[str],
    used_worker_ids:  set[str],
) -> tuple[list[RouteOutput], list[UnscheduledManifest], set[str], set[str]]:
    """
    Hub-to-branch routing.

    Each hub_to_branch worker has `assignedBranches` = the subset of local
    branches they serve from the hub.

    Strategy:
      1. For each worker, filter manifests to only those whose
         destinationBranchId is in that worker's assignedBranches.
      2. Among all workers, find the best assignment of manifests to
         workers+vehicles using a lightweight greedy approach
         (GA would be overkill here — the load unit is already a manifest,
         not a package, so there are far fewer items to assign).
      3. For each worker's manifest set, build an ordered multi-stop route
         using nearest-neighbour + 2-opt on branch coordinates.

    If a manifest's destination branch is not covered by any worker's
    assignedBranches, it is marked unscheduled.
    """
    routes:      list[RouteOutput]         = []
    unscheduled: list[UnscheduledManifest] = []

    newly_used_vehicle_ids: set[str] = set()
    newly_used_worker_ids:  set[str] = set()

    available_workers  = [w for w in workers if w.id not in used_worker_ids]
    available_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]

    if not available_workers or not available_vehicles:
        for m in manifests:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason="No hub_to_branch workers or vehicles available",
            ))
        return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids

    # Build a reverse index: branchId → workers who serve it
    branch_to_workers: dict[str, list[WorkerInput]] = {}
    for w in available_workers:
        for branch_id in (w.assignedBranches or []):
            branch_to_workers.setdefault(branch_id, []).append(w)

    # Check coverage: mark manifests whose destination has no covering worker
    covered_manifests:   list[ManifestInput] = []
    for m in manifests:
        if m.destinationBranchId in branch_to_workers:
            covered_manifests.append(m)
        else:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason=f"No hub_to_branch worker covers branch {m.destinationBranchId}",
            ))

    if not covered_manifests:
        return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids

    # ── Greedy manifest→worker assignment ────────────────────────────────────
    # Build a per-worker manifest list: each manifest goes to the worker whose
    # assignedBranches contains its destination AND who has the most remaining
    # capacity (greedy fill-first to minimise vehicles used).

    worker_manifest_map: dict[str, list[ManifestInput]] = {
        w.id: [] for w in available_workers
    }
    worker_weight_used:  dict[str, float] = {w.id: 0.0 for w in available_workers}
    worker_volume_used:  dict[str, float] = {w.id: 0.0 for w in available_workers}

    # Pick a vehicle per worker (tentative; we re-check capacity later)
    worker_vehicle: dict[str, VehicleInput | None] = {w.id: None for w in available_workers}
    remaining_vehicles = list(available_vehicles)

    for w in available_workers:
        veh = _pick_vehicle_for_worker(remaining_vehicles, newly_used_vehicle_ids)
        if veh:
            worker_vehicle[w.id] = veh
            remaining_vehicles = [v for v in remaining_vehicles if v.id != veh.id]

    # Sort manifests: heaviest first
    covered_manifests.sort(key=lambda m: -m.totalWeight)

    for m in covered_manifests:
        dest = m.destinationBranchId
        candidate_workers = [
            w for w in available_workers
            if dest in (w.assignedBranches or [])
            and worker_vehicle.get(w.id) is not None
        ]
        if not candidate_workers:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason=f"No worker with vehicle available for branch {dest}",
            ))
            continue

        # Pick worker with highest remaining capacity (weight-based tiebreak)
        def remaining_cap(w: WorkerInput) -> float:
            veh = worker_vehicle[w.id]
            return veh.maxWeight * CAPACITY_BUFFER - worker_weight_used[w.id]

        candidate_workers.sort(key=remaining_cap, reverse=True)
        assigned = False
        for w in candidate_workers:
            veh = worker_vehicle[w.id]
            new_w = worker_weight_used[w.id] + m.totalWeight
            new_v = worker_volume_used[w.id] + m.totalVolume
            if (
                new_w <= veh.maxWeight * CAPACITY_BUFFER
                and new_v <= veh.maxVolume * CAPACITY_BUFFER
            ):
                worker_manifest_map[w.id].append(m)
                worker_weight_used[w.id] = new_w
                worker_volume_used[w.id] = new_v
                assigned = True
                break

        if not assigned:
            unscheduled.append(UnscheduledManifest(
                manifestId=m.id,
                reason="Capacity exceeded on all workers that serve this branch",
            ))

    # ── Build routes for each worker ─────────────────────────────────────────
    for worker in available_workers:
        worker_manifests = worker_manifest_map.get(worker.id, [])
        if not worker_manifests:
            continue  # nothing assigned to this worker

        veh = worker_vehicle.get(worker.id)
        if veh is None:
            for m in worker_manifests:
                unscheduled.append(UnscheduledManifest(
                    manifestId=m.id, reason="No vehicle assigned to worker"
                ))
            continue

        # Build stop points: one stop per destination branch
        stops = _build_manifest_stops(worker_manifests)

        # Build OSRM or Haversine distance matrix for stop ordering
        all_stop_coords = [s.coords for s in stops]
        global_dm = GlobalDistanceMatrix.build(origin, all_stop_coords)

        stop_index_map: dict[str, int] = {}
        sub_matrix: list[list[float]] | None = None
        if global_dm.source == "osrm":
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
            route_type="inter_branch",
            dist_matrix=sub_matrix,
            stop_index_map=stop_index_map if sub_matrix else None,
        )

        total_drive = sum(route_result.segment_drive_minutes)
        total_dwell = len(stops) * HUB_BRANCH_DWELL
        total_time  = total_drive + total_dwell

        # Build StopOutput list (ordered by the optimised route)
        stop_outputs: list[StopOutput] = []
        manifest_by_stop: dict[str, list[ManifestInput]] = {}
        for m in worker_manifests:
            manifest_by_stop.setdefault(m.destinationBranchId, []).append(m)

        for stop_pt in route_result.ordered_stops:
            branch_id = stop_pt.meta.get("destinationBranchId", stop_pt.id)
            manifests_at_stop = manifest_by_stop.get(branch_id, [])
            stop_outputs.append(StopOutput(
                coordinates=stop_pt.coords,
                manifestIds=[m.id for m in manifests_at_stop],
                destinationBranchId=branch_id,
            ))

        distance_km = round(route_result.total_distance_km, 2) if global_dm.source != "n/a" else 0.0
        distance_source = route_result.distance_source if global_dm.source != "n/a" else "n/a"

        route = RouteOutput(
            vehicleId=veh.id,
            workerId=worker.id,
            routeType="hub_to_branch",
            stops=stop_outputs,
            packageIds=[],
            manifestIds=[m.id for m in worker_manifests],
            totalWeight=round(worker_weight_used[worker.id], 2),
            totalVolume=round(worker_volume_used[worker.id], 4),
            distanceKm=distance_km,
            estimatedTimeMinutes=total_time,
            distanceSource=distance_source,
        )
        routes.append(route)

        newly_used_vehicle_ids.add(veh.id)
        newly_used_worker_ids.add(worker.id)

        logger.debug(
            f"[pipeline] [hub_to_branch] vehicle={veh.registrationNumber} "
            f"worker={worker.id} manifests={len(worker_manifests)} "
            f"stops={len(stops)} dist={route_result.total_distance_km:.1f}km "
            f"time={total_time}min"
        )

    return routes, unscheduled, newly_used_vehicle_ids, newly_used_worker_ids


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