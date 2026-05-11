from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.telemetry.metrics import MetricsMiddleware, metrics_response


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.metrics_enabled:
    app.add_middleware(MetricsMiddleware)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/metrics")
async def metrics() -> object:
    return metrics_response()

