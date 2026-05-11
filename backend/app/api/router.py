from fastapi import APIRouter

from app.api.routes import anomalies, experiments, health, models, predict, recommend, scheduler

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, prefix="/predict", tags=["prediction"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["recommendation"])
api_router.include_router(experiments.router, tags=["experiments"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(anomalies.router, tags=["anomalies"])
api_router.include_router(scheduler.router, tags=["scheduler"])

