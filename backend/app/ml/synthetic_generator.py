from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from random import Random
from uuid import uuid4

import numpy as np
import pandas as pd


MODEL_TYPES = [
    "resnet",
    "bert",
    "gpt",
    "vision_transformer",
    "stable_diffusion",
    "recommendation",
]
FRAMEWORKS = ["pytorch", "tensorflow", "jax"]
GPU_TYPES = ["t4", "v100", "a100", "rtx4090"]
FAILURE_REASONS = ["oom", "cuda_error", "nan_loss", "data_loader_stall", "none"]


@dataclass
class SyntheticDataConfig:
    num_samples: int = 50_000
    seed: int = 42
    start_time: datetime = datetime(2025, 1, 1)
    output_path: str = "data/synthetic_jobs.csv"


class SyntheticWorkloadGenerator:
    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self.rng = Random(config.seed)
        np.random.seed(config.seed)

    def generate(self) -> pd.DataFrame:
        records: list[dict] = []
        for _ in range(self.config.num_samples):
            model_type = self.rng.choice(MODEL_TYPES)
            framework = self.rng.choice(FRAMEWORKS)
            gpu_type = self.rng.choice(GPU_TYPES)
            num_gpus = self._sample_num_gpus(model_type)
            batch_size = self._sample_batch_size(model_type)
            dataset_size = self.rng.randint(10_000, 10_000_000)
            cpu_cores = self.rng.choice([8, 16, 24, 32, 48, 64, 96])
            ram_gb = self.rng.choice([16, 32, 64, 128, 256, 512])
            distributed = num_gpus > 1 and self.rng.random() < 0.85
            mixed_precision = self.rng.random() < 0.7
            estimated_flops = self._estimate_flops(model_type, dataset_size, batch_size)

            runtime_hours = self._estimate_runtime_hours(
                model_type=model_type,
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                batch_size=batch_size,
                dataset_size=dataset_size,
                estimated_flops=estimated_flops,
                distributed_training=distributed,
                mixed_precision=mixed_precision,
            )

            peak_vram = self._estimate_peak_vram(
                model_type=model_type,
                gpu_type=gpu_type,
                batch_size=batch_size,
                num_gpus=num_gpus,
                mixed_precision=mixed_precision,
            )

            gpu_util = float(np.clip(np.random.normal(loc=76, scale=12), 20, 99))
            failed, reason = self._sample_failure(
                model_type=model_type,
                gpu_type=gpu_type,
                batch_size=batch_size,
                peak_vram=peak_vram,
                num_gpus=num_gpus,
                distributed_training=distributed,
            )

            timestamp = self.config.start_time + timedelta(minutes=self.rng.randint(0, 60 * 24 * 365))
            records.append(
                {
                    "job_id": str(uuid4()),
                    "timestamp": timestamp.isoformat(),
                    "model_type": model_type,
                    "framework": framework,
                    "gpu_type": gpu_type,
                    "num_gpus": num_gpus,
                    "batch_size": batch_size,
                    "dataset_size": dataset_size,
                    "cpu_cores": cpu_cores,
                    "ram_gb": ram_gb,
                    "estimated_flops": round(estimated_flops, 2),
                    "distributed_training": distributed,
                    "mixed_precision": mixed_precision,
                    "runtime_hours": round(runtime_hours, 3),
                    "peak_vram_gb": round(peak_vram, 3),
                    "gpu_utilization": round(gpu_util, 3),
                    "failed": failed,
                    "failure_reason": reason,
                }
            )
        return pd.DataFrame(records)

    def export_csv(self, frame: pd.DataFrame) -> Path:
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output

    def _sample_num_gpus(self, model_type: str) -> int:
        if model_type in {"gpt", "stable_diffusion"}:
            return self.rng.choice([1, 2, 4, 8])
        if model_type in {"bert", "vision_transformer"}:
            return self.rng.choice([1, 2, 4])
        return self.rng.choice([1, 2])

    def _sample_batch_size(self, model_type: str) -> int:
        if model_type in {"gpt", "stable_diffusion"}:
            return self.rng.choice([8, 16, 32, 64, 128])
        return self.rng.choice([16, 32, 64, 128, 256, 512])

    def _estimate_flops(self, model_type: str, dataset_size: int, batch_size: int) -> float:
        multipliers = {
            "resnet": 8e8,
            "bert": 2.5e9,
            "gpt": 1.2e10,
            "vision_transformer": 4.5e9,
            "stable_diffusion": 1.5e10,
            "recommendation": 7e8,
        }
        return multipliers[model_type] * (dataset_size / batch_size)

    def _estimate_runtime_hours(
        self,
        model_type: str,
        gpu_type: str,
        num_gpus: int,
        batch_size: int,
        dataset_size: int,
        estimated_flops: float,
        distributed_training: bool,
        mixed_precision: bool,
    ) -> float:
        gpu_speed = {"t4": 0.9, "v100": 1.4, "a100": 2.5, "rtx4090": 2.1}[gpu_type]
        model_factor = {
            "resnet": 0.8,
            "bert": 1.2,
            "gpt": 2.8,
            "vision_transformer": 1.6,
            "stable_diffusion": 2.4,
            "recommendation": 0.9,
        }[model_type]

        base = (estimated_flops / 1e14) * model_factor / (gpu_speed * max(num_gpus, 1))
        io_penalty = np.log10(dataset_size) * 0.1
        distributed_penalty = 0.92 if distributed_training else 1.0
        mp_boost = 0.88 if mixed_precision else 1.0
        noise = np.random.normal(1.0, 0.08)
        return float(max(0.05, base * io_penalty * distributed_penalty * mp_boost * noise))

    def _estimate_peak_vram(
        self,
        model_type: str,
        gpu_type: str,
        batch_size: int,
        num_gpus: int,
        mixed_precision: bool,
    ) -> float:
        model_base = {
            "resnet": 6.0,
            "bert": 12.0,
            "gpt": 28.0,
            "vision_transformer": 18.0,
            "stable_diffusion": 24.0,
            "recommendation": 8.0,
        }[model_type]
        gpu_adjust = {"t4": 1.12, "v100": 1.0, "a100": 0.9, "rtx4090": 0.95}[gpu_type]
        per_gpu_batch = batch_size / max(num_gpus, 1)
        memory_from_batch = np.sqrt(per_gpu_batch) * 1.6
        precision_factor = 0.82 if mixed_precision else 1.0
        return float((model_base + memory_from_batch) * gpu_adjust * precision_factor)

    def _sample_failure(
        self,
        model_type: str,
        gpu_type: str,
        batch_size: int,
        peak_vram: float,
        num_gpus: int,
        distributed_training: bool,
    ) -> tuple[bool, str]:
        vram_capacity = {"t4": 16, "v100": 32, "a100": 80, "rtx4090": 24}[gpu_type]
        prob = 0.03
        if peak_vram > vram_capacity:
            prob += 0.45
        if model_type in {"gpt", "stable_diffusion"} and gpu_type == "t4":
            prob += 0.12
        if batch_size >= 256:
            prob += 0.1
        if distributed_training and num_gpus >= 4:
            prob += 0.04

        failed = self.rng.random() < min(prob, 0.9)
        if not failed:
            return False, "none"

        if peak_vram > vram_capacity:
            return True, "oom"
        reason_weights = [0.2, 0.35, 0.25, 0.2]
        reason = np.random.choice(FAILURE_REASONS[:-1], p=reason_weights)
        return True, str(reason)

