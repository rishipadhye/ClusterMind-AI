from fastapi import APIRouter

from app.models.schemas import JobMetadata, ResourceRecommendationResponse
from app.services.prediction_service import PredictionService
from app.services.recommendation_service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService(prediction_service=PredictionService())


@router.post("/resources", response_model=ResourceRecommendationResponse)
async def recommend_resources(payload: JobMetadata) -> ResourceRecommendationResponse:
    return recommendation_service.recommend(payload)

