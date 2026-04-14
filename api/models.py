# ─────────────────────────────────────────────────────────────────────────────
#  api/models.py
#  Pydantic v2 request / response models.
#  Field names and shapes mirror the TypeScript types in types.util.ts so
#  Node.js can serialize its internal objects and send them with minimal
#  transformation.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Coordinates ───────────────────────────────────────────────────────────────
# GeoJSON order: [longitude, latitude]  (matches Node.js Coords type)
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

    model_config = {"populate_by_name": True}


class OptimizeRequest(BaseModel):
    """
    Full payload sent by Node.js orchestrator.
    Contains ALL packages for one branch (both transporter + deliverer).
    Python splits them internally by deliveryType / destinationBranchId.
    """
    branch: BranchInput
    vehicles: list[VehicleInput]
    workers: list[WorkerInput]
    packages: list[PackageInput]


# ── Output models ─────────────────────────────────────────────────────────────

class StopOutput(BaseModel):
    coordinates: Coords
    packageIds: list[str]
    # Only present on deliverer stops
    address: Optional[str] = None
    recipientName: Optional[str] = None
    # Only present on transporter stops
    destinationBranchId: Optional[str] = None


class RouteOutput(BaseModel):
    vehicleId: str
    workerId: str
    routeType: Literal["inter_branch", "local_delivery"]
    stops: list[StopOutput]
    packageIds: list[str]
    totalWeight: float
    totalVolume: float
    distanceKm: float
    estimatedTimeMinutes: int
    # "osrm"      — real road distances from OSRM (deliverer routes)
    # "haversine" — straight-line fallback (deliverer routes when OSRM unavailable)
    # "n/a"       — inter-branch routes: distance is 0 because branch coordinates
    #               are not in the request; Node.js resolves them at persist time
    distanceSource: Literal["osrm", "haversine", "n/a"]


class UnscheduledPackage(BaseModel):
    packageId: str
    reason: str


class OptimizeResponse(BaseModel):
    routes: list[RouteOutput]
    unscheduled: list[UnscheduledPackage]
    meta: dict  # timing, package counts, etc.