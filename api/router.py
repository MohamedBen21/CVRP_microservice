# ─────────────────────────────────────────────────────────────────────────────
#  api/router.py
#  FastAPI route definitions.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, HTTPException
from api.models import OptimizeRequest, OptimizeResponse
from services.optimizer_pipeline import run_optimization
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(payload: OptimizeRequest) -> OptimizeResponse:

    """
    Full optimization for one branch or hub.

    Handles three separate workloads in one call:
      • hub_to_hub workers    → manifest-based direct hub legs
      • hub_to_branch workers → manifest-based multi-stop branch runs
      • deliverers            → package-based last-mile routes

    Node.js sends whichever combination is relevant for the given branch/hub.
    """
    
    t0 = time.perf_counter()

    has_packages  = bool(payload.packages)
    has_manifests = bool(payload.manifests)

    if not has_packages and not has_manifests:
        return OptimizeResponse(
            routes=[],
            unscheduled=[],
            unscheduledManifests=[],
            meta={"durationMs": 0},
        )

    if not payload.vehicles:
        return OptimizeResponse(
            routes=[],
            unscheduled=[
                {"packageId": p.id, "reason": "No vehicles available"}
                for p in payload.packages
            ],

            unscheduledManifests=[
                {"manifestId": m.id, "reason": "No vehicles available"}
                for m in payload.manifests
            ],

            meta={"durationMs": 0},
        )

    if not payload.workers:
        return OptimizeResponse(
            routes=[],
            unscheduled=[
                {"packageId": p.id, "reason": "No workers available"}
                for p in payload.packages
            ],

            unscheduledManifests=[
                {"manifestId": m.id, "reason": "No workers available"}
                for m in payload.manifests
            ],

            meta={"durationMs": 0},
        )

    try:
        result = run_optimization(payload)
    except Exception as exc:
        logger.exception("Optimization pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    result.meta["durationMs"] = elapsed_ms

    logger.info(
        f"[optimize] branch={payload.branch.id} "
        f"pkg={len(payload.packages)} manifests={len(payload.manifests)} "
        f"routes={len(result.routes)} "
        f"unscheduled_pkg={len(result.unscheduled)} "
        f"unscheduled_man={len(result.unscheduledManifests)} "
        f"time={elapsed_ms}ms"
    )

    return result