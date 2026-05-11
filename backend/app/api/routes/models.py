from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.settings import get_settings

router = APIRouter()


@router.get("/models/status")
async def model_status() -> dict:
    settings = get_settings()
    artifact = Path(settings.model_bundle_path)
    metrics = Path("data/models/metrics.json")
    return {
        "artifact_exists": artifact.exists(),
        "metrics_exists": metrics.exists(),
        "artifact_path": str(artifact),
        "metrics_path": str(metrics),
    }

