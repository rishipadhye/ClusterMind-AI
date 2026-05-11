from app.models.schemas import JobMetadata, ResourceRecommendationResponse
from app.services.prediction_service import PredictionService


class RecommendationService:
    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def recommend(self, job: JobMetadata) -> ResourceRecommendationResponse:
        runtime, runtime_conf = self.prediction_service.predict_runtime(job)
        vram, vram_conf = self.prediction_service.predict_vram(job)
        failure_prob, failure_conf, _ = self.prediction_service.predict_failure_probability(job)

        suggested_gpus = job.num_gpus
        suggestions: list[str] = []
        if failure_prob > 0.3 and job.batch_size > 64:
            suggestions.append(
                f"Reduce batch size from {job.batch_size} to {max(job.batch_size // 2, 32)} to lower OOM risk."
            )
        if runtime > 6:
            suggested_gpus = min(job.num_gpus + 1, 16)
            suggestions.append("Increase GPU count for better runtime predictability.")
        if not job.mixed_precision:
            suggestions.append("Enable mixed precision for memory and throughput gains.")
        if job.distributed_training and job.num_gpus == 1:
            suggestions.append("Enable distributed training for multi-GPU scaling efficiency.")
        if runtime_conf < 0.65 or vram_conf < 0.65 or failure_conf < 0.65:
            suggestions.append("Prediction uncertainty is high; collect more telemetry for this workload type.")

        if not suggestions:
            suggestions.append("Current configuration is close to optimal for this workload profile.")

        return ResourceRecommendationResponse(
            recommended_gpu_count=suggested_gpus,
            recommended_vram_gb=round(vram * 1.1, 2),
            expected_runtime_hours=round(runtime, 2),
            failure_risk=round(failure_prob, 3),
            optimization_suggestions=suggestions,
        )

