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
    Full CVRP optimization for one branch.

    Node.js sends raw packages + vehicles + workers.
    Python returns ready-to-persist routes with ordered stops.
    """
    t0 = time.perf_counter()

    if not payload.packages:
        return OptimizeResponse(routes=[], unscheduled=[], meta={"durationMs": 0})

    if not payload.vehicles:
        return OptimizeResponse(
            routes=[],
            unscheduled=[
                {"packageId": p.id, "reason": "No vehicles available"}
                for p in payload.packages
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
        f"pkg={len(payload.packages)} "
        f"routes={len(result.routes)} "
        f"unscheduled={len(result.unscheduled)} "
        f"time={elapsed_ms}ms"
    )

    return result
