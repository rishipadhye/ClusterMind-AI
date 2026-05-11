import logging

import numpy as np

from app.ml.model_store import ModelStore
from app.models.schemas import JobMetadata


class PredictionService:
    model_version = "heuristic-v0"

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_store = ModelStore()

    def predict_runtime(self, job: JobMetadata) -> tuple[float, float]:
        bundle = self.model_store.load_bundle()
        if bundle:
            features = self.model_store.to_features(job)[bundle["feature_columns"]]
            transformed = bundle["preprocessor"].transform(features)
            pred = float(bundle["runtime_model"].predict(transformed)[0])
            std = float(bundle.get("target_stats", {}).get("runtime", {}).get("std", 1.0))
            rmse = float(bundle["metrics"]["runtime"].get("rmse", 1.0))
            confidence = self._regression_confidence(rmse=rmse, target_std=std)
            self.model_version = bundle["version"]
            return max(pred, 0.01), confidence

        baseline = (job.dataset_size / max(job.batch_size * job.num_gpus, 1)) * 0.02
        complexity_boost = 1.8 if job.model_type in {"gpt", "stable_diffusion"} else 1.0
        runtime_hours = max(0.1, baseline * complexity_boost)
        confidence = 0.62
        return runtime_hours, confidence

    def predict_vram(self, job: JobMetadata) -> tuple[float, float]:
        bundle = self.model_store.load_bundle()
        if bundle:
            features = self.model_store.to_features(job)[bundle["feature_columns"]]
            transformed = bundle["preprocessor"].transform(features)
            pred = float(bundle["vram_model"].predict(transformed)[0])
            std = float(bundle.get("target_stats", {}).get("vram", {}).get("std", 1.0))
            rmse = float(bundle["metrics"]["vram"].get("rmse", 1.0))
            confidence = self._regression_confidence(rmse=rmse, target_std=std)
            self.model_version = bundle["version"]
            return max(pred, 0.1), confidence

        vram = (job.batch_size * 0.06) + (job.estimated_flops / 1e14) + 2.0
        if job.model_type in {"gpt", "vision_transformer"}:
            vram += 4.0
        confidence = 0.64
        return vram, confidence

    def predict_failure_probability(self, job: JobMetadata) -> tuple[float, float, str]:
        bundle = self.model_store.load_bundle()
        if bundle:
            features = self.model_store.to_features(job)[bundle["feature_columns"]]
            transformed = bundle["preprocessor"].transform(features)
            probability = float(bundle["failure_model"].predict_proba(transformed)[0][1])
            roc_auc = float(bundle["metrics"]["failure"].get("roc_auc", 0.65))
            confidence = float(np.clip((roc_auc - 0.5) / 0.5, 0.5, 0.95))
            threshold = float(bundle.get("failure_threshold", bundle["metrics"]["failure"].get("threshold", 0.5)))
            self.model_version = bundle["version"]
            reason = "oom_likely" if probability >= max(0.35, threshold) else "none"
            return probability, confidence, reason

        risk = 0.05
        if job.batch_size >= 512:
            risk += 0.15
        if job.ram_gb < 16:
            risk += 0.10
        if job.model_type in {"gpt", "stable_diffusion"} and job.gpu_type == "t4":
            risk += 0.18
        reason = "oom_likely" if risk > 0.2 else "none"
        return min(risk, 0.95), 0.61, reason

    def _regression_confidence(self, rmse: float, target_std: float) -> float:
        # Normalized RMSE: rmse/std. Map lower error => higher confidence.
        denom = max(target_std, 1e-6)
        nrmse = rmse / denom
        confidence = 1.0 - min(1.0, nrmse)
        return float(np.clip(confidence, 0.5, 0.95))
