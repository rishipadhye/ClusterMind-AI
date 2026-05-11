from fastapi import FastAPI

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

if settings.metrics_enabled:
    app.add_middleware(MetricsMiddleware)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/metrics")
async def metrics() -> object:
    return metrics_response()

