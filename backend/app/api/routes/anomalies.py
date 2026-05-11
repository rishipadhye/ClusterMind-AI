from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from fastapi import APIRouter

from app.core.settings import get_settings
from app.ml.model_store import ModelStore
from app.models.schemas import JobMetadata

router = APIRouter()


@router.post("/anomalies/score")
async def anomaly_score(payload: JobMetadata) -> dict[str, Any]:
    settings = get_settings()
    artifact = Path(settings.anomaly_bundle_path)
    if not artifact.exists():
        return {"artifact_loaded": False, "score": None, "is_anomaly": None}

    bundle = joblib.load(artifact)
    store = ModelStore()
    features = store.to_features(payload)[bundle["feature_columns"]]
    transformed = bundle["preprocessor"].transform(features)

    # IsolationForest: lower score => more anomalous
    score = float(bundle["model"].score_samples(transformed)[0])
    is_anomaly = bool(bundle["model"].predict(transformed)[0] == -1)
    return {"artifact_loaded": True, "score": score, "is_anomaly": is_anomaly, "version": bundle.get("version")}
