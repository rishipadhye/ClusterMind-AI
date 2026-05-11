from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import JobMetadata
from app.scheduler.simulator import SchedulerSimulator
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/scheduler/simulate")
async def simulate_scheduler(payloads: list[JobMetadata]) -> dict:
    predictor = PredictionService()
    sim = SchedulerSimulator(predictor)
    result = sim.simulate(payloads)
    return {
        "avg_reward": result.avg_reward,
        "avg_runtime_hours": result.avg_runtime_hours,
        "avg_failure_prob": result.avg_failure_prob,
        "allocation_histogram": result.allocation_histogram,
    }
