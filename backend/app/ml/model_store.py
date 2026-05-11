from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.settings import get_settings
from app.ml.features import add_engineered_features
from app.models.schemas import JobMetadata


class ModelStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.bundle_path = Path(settings.model_bundle_path)
        self._bundle: dict[str, Any] | None = None

    def load_bundle(self) -> dict[str, Any] | None:
        if self._bundle is not None:
            return self._bundle
        if not self.bundle_path.exists():
            return None
        self._bundle = joblib.load(self.bundle_path)
        return self._bundle

    def to_features(self, job: JobMetadata) -> pd.DataFrame:
        raw = pd.DataFrame(
            [
                {
                    "model_type": job.model_type,
                    "framework": job.framework,
                    "gpu_type": job.gpu_type,
                    "num_gpus": job.num_gpus,
                    "batch_size": job.batch_size,
                    "dataset_size": job.dataset_size,
                    "cpu_cores": job.cpu_cores,
                    "ram_gb": job.ram_gb,
                    "estimated_flops": job.estimated_flops,
                    "distributed_training": int(job.distributed_training),
                    "mixed_precision": int(job.mixed_precision),
                    "gpu_utilization": 75.0,
                }
            ]
        )
        enriched = add_engineered_features(raw)
        return enriched

