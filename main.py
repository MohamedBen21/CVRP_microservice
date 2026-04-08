# ─────────────────────────────────────────────────────────────────────────────
#  main.py
#  FastAPI application entry point.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from api.router import router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    osrm_url = os.getenv("OSRM_URL", "http://localhost:5000")
    logger.info(f"Optimizer starting up — OSRM_URL={osrm_url}")
    yield
    logger.info("Optimizer shutting down")


app = FastAPI(
    title="Logistics Route Optimizer",
    description="CVRP optimization service for delivery route planning",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
