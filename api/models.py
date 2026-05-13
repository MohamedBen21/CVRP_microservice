# ─────────────────────────────────────────────────────────────────────────────
#  api/models.py
#  Pydantic v2 request / response models.
#  Field names and shapes mirror the TypeScript types so Node.js can serialize
#  its internal objects with minimal transformation.
#
#  Hub model additions
#  ────────────────────
#  Transporters now come in two sub-types:
#
#    hub_to_hub    → WorkerInput.transporterType == "hub_to_hub"
#                   The optimizer produces a single 2-stop route (origin hub →
#                   destination hub).  No GA needed — it is purely a direct leg.
#                   The request carries ManifestInput objects instead of raw
#                   PackageInput objects for this transporter type.
#
#    hub_to_branch → WorkerInput.transporterType == "hub_to_branch"
#                   Multi-stop route across assignedBranches, each stop receives
#                   one or more manifests (sealed bags).
#                   GA + nearest-neighbour + 2-opt still runs, but the unit of
#                   load is now a manifest (weight = sum of packages inside),
#                   not an individual package.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Coordinates ───────────────────────────────────────────────────────────────
# GeoJSON order: [longitude, latitude]
Coords = tuple[float, float]


# ── Input models ──────────────────────────────────────────────────────────────

class BranchInput(BaseModel):
    id: str = Field(..., alias="_id")
    coordinates: Coords          # [lng, lat]

    model_config = {"populate_by_name": True}


class PackageDestination(BaseModel):
    coordinates: Coords
    recipientName: str = ""
    recipientPhone: str = ""
    address: str = ""
    city: str = ""
    state: str = ""


class PackageInput(BaseModel):
    id: str = Field(..., alias="_id")
    weight: float                # kg
    volume: float                # m³
    isFragile: bool = False
    deliveryType: Literal["home", "branch_pickup"]
    deliveryPriority: Literal["standard", "express", "same_day"] = "standard"

    # Transporter packages: which branch this package travels to next
    destinationBranchId: Optional[str] = None
    # Deliverer packages: where to physically drop it off
    destination: Optional[PackageDestination] = None

    model_config = {"populate_by_name": True}


class ManifestInput(BaseModel):
    """
    A sealed manifest bag carried by a transporter.
    This is the load unit for hub-model transporter routes — the optimizer
    treats each manifest the same way it used to treat a single package,
    but the weight/volume are the aggregate of all packages inside the bag.
    """
    id: str = Field(..., alias="_id")
    manifestCode: str

    # Physical properties of the sealed bag
    totalWeight: float           # kg — sum of all packages inside
    packageCount: int            # number of packages (informational)
    # Volume is optional; use 0.0 when not tracked at manifest level
    totalVolume: float = 0.0     # m³

    # Where this manifest originates from (its current location)
    originBranchId: str
    # Coordinates of the origin branch (resolved by Node.js before the call)
    originCoordinates: Coords

    # Where this manifest needs to be delivered / dropped off
    destinationBranchId: str
    # Coordinates of the destination branch (resolved by Node.js before the call)
    destinationCoordinates: Coords

    priority: Literal["standard", "express", "urgent"] = "standard"

    model_config = {"populate_by_name": True}


class VehicleInput(BaseModel):
    id: str = Field(..., alias="_id")
    type: Literal["motorcycle", "car", "van", "small_truck", "large_truck"]
    maxWeight: float
    maxVolume: float
    supportsFragile: bool = True
    registrationNumber: str

    model_config = {"populate_by_name": True}


class WorkerInput(BaseModel):
    id: str = Field(..., alias="_id")
    userId: str
    role: Literal["transporter", "deliverer"]

    # Hub model extension — optional; omit for legacy transporters / deliverers
    transporterType: Optional[Literal["hub_to_hub", "hub_to_branch"]] = None

    # hub_to_hub: the two hub branch IDs this transporter shuttles between.
    # Always [originHubId, destinationHubId] from the perspective of this trip.
    assignedLine: Optional[list[str]] = None

    # hub_to_branch: the branch IDs this transporter serves from their home hub.
    # The optimizer will build stops only for branches present in this list.
    assignedBranches: Optional[list[str]] = None

    model_config = {"populate_by_name": True}


class OptimizeRequest(BaseModel):
    """
    Full payload sent by Node.js orchestrator for one hub/branch.

    Node.js is responsible for splitting and pre-loading the correct data:
      • For hub_to_hub workers  → send `manifests` only (packages list empty).
      • For hub_to_branch workers → send `manifests` only (packages list empty).
      • For deliverers           → send `packages` only (manifests list empty).
      • Mixed hubs               → can send both; Python splits by worker role.

    Python separates the three workloads internally and runs independent
    optimization passes for each.
    """
    branch: BranchInput
    vehicles: list[VehicleInput]
    workers: list[WorkerInput]

    # Raw packages — used for deliverer pass only
    packages: list[PackageInput] = Field(default_factory=list)

    # Manifests — used for hub_to_hub and hub_to_branch transporter passes
    manifests: list[ManifestInput] = Field(default_factory=list)


# ── Output models ─────────────────────────────────────────────────────────────

class StopOutput(BaseModel):
    coordinates: Coords
    packageIds: list[str] = Field(default_factory=list)

    # Present on deliverer stops
    address: Optional[str] = None
    recipientName: Optional[str] = None

    # Present on transporter stops (hub_to_hub and hub_to_branch)
    destinationBranchId: Optional[str] = None
    # Manifest IDs loaded/unloaded at this stop (hub model)
    manifestIds: list[str] = Field(default_factory=list)


class RouteOutput(BaseModel):
    vehicleId: str
    workerId: str
    routeType: Literal["hub_to_hub", "hub_to_branch", "local_delivery"]
    stops: list[StopOutput]
    # For hub_to_hub routes: which hub this leg departs from.
    # Node.js uses this to set originBranchId correctly on the persisted route
    # (critical for return legs which originate at hub B, not hub A).
    originBranchId: Optional[str] = None

    # For package-based routes (deliverer / legacy transporter)
    packageIds: list[str] = Field(default_factory=list)
    # For manifest-based routes (hub model transporters)
    manifestIds: list[str] = Field(default_factory=list)

    totalWeight: float
    totalVolume: float
    distanceKm: float
    estimatedTimeMinutes: int

    # "osrm"      — real road distances from OSRM
    # "haversine" — straight-line fallback
    # "n/a"       — distance placeholder (resolved by Node.js at persist time)
    distanceSource: Literal["osrm", "haversine", "n/a"]


class UnscheduledPackage(BaseModel):
    packageId: str
    reason: str


class UnscheduledManifest(BaseModel):
    manifestId: str
    reason: str


class OptimizeResponse(BaseModel):
    routes: list[RouteOutput]
    unscheduled: list[UnscheduledPackage] = Field(default_factory=list)
    unscheduledManifests: list[UnscheduledManifest] = Field(default_factory=list)
    meta: dict  # timing, counts, etc.