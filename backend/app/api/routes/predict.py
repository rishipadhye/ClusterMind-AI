from fastapi import APIRouter

from app.models.schemas import FailurePredictionResponse, JobMetadata, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()


@router.post("/runtime", response_model=PredictionResponse)
async def predict_runtime(payload: JobMetadata) -> PredictionResponse:
    value, confidence = prediction_service.predict_runtime(payload)
    return PredictionResponse(
        value=round(value, 4),
        confidence=confidence,
        model_version=prediction_service.model_version,
    )


@router.post("/vram", response_model=PredictionResponse)
async def predict_vram(payload: JobMetadata) -> PredictionResponse:
    value, confidence = prediction_service.predict_vram(payload)
    return PredictionResponse(
        value=round(value, 4),
        confidence=confidence,
        model_version=prediction_service.model_version,
    )


@router.post("/failure", response_model=FailurePredictionResponse)
async def predict_failure(payload: JobMetadata) -> FailurePredictionResponse:
    probability, confidence, reason = prediction_service.predict_failure_probability(payload)
    return FailurePredictionResponse(
        probability=round(probability, 4),
        confidence=confidence,
        model_version=prediction_service.model_version,
        likely_reason=reason,
    )

