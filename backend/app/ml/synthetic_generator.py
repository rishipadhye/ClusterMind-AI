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
        # Batch ranges reflect what practitioners actually use per family: memory-
        # hungry generative models run small batches (often with grad accumulation),
        # while CNN/recommendation models scale to large batches.
        if model_type == "stable_diffusion":
            return self.rng.choice([1, 2, 4, 8, 16])
        if model_type == "gpt":
            return self.rng.choice([4, 8, 16, 32, 64])
        if model_type in {"bert", "vision_transformer"}:
            return self.rng.choice([8, 16, 32, 64, 128])
        return self.rng.choice([16, 32, 64, 128, 256, 512])

    def _estimate_flops(self, model_type: str, dataset_size: int, batch_size: int) -> float:
        # Total training compute scales with FLOPs-per-sample times the number of
        # samples processed (one epoch proxy). It is essentially independent of the
        # batch size -- batching changes throughput/parallelism, not the total work
        # done -- so we do NOT divide by batch here (an earlier version did, which
        # made tiny-batch generative jobs report physically impossible FLOP counts).
        flops_per_sample = {
            "resnet": 8e8,
            "bert": 2.5e9,
            "gpt": 1.2e10,
            "vision_transformer": 4.5e9,
            "stable_diffusion": 1.5e10,
            "recommendation": 7e8,
        }
        return flops_per_sample[model_type] * dataset_size

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

        # Runtime = total compute / effective throughput. Throughput scales with the
        # GPU's relative speed and the GPU count (with a sub-linear scaling penalty
        # for distributed training). Mixed precision gives a real wall-clock speedup.
        effective_throughput = 3e14 * gpu_speed * max(num_gpus, 1)
        scaling_penalty = 1.12 if (distributed_training and num_gpus >= 4) else 1.0
        base = estimated_flops * model_factor / effective_throughput * scaling_penalty
        mp_boost = 0.7 if mixed_precision else 1.0
        noise = np.random.normal(1.0, 0.08)
        # Clip to a realistic single-job window (a few minutes to ~2 weeks).
        return float(np.clip(base * mp_boost * noise, 0.05, 400.0))

    def _estimate_peak_vram(
        self,
        model_type: str,
        gpu_type: str,
        batch_size: int,
        num_gpus: int,
        mixed_precision: bool,
    ) -> float:
        # Physically-grounded peak-VRAM model (see README "Synthetic generator
        # assumptions"): peak memory ~= a fixed cost for weights + optimizer state,
        # plus activation memory that grows ~linearly with the per-GPU batch size.
        #
        #   peak = weights_optimizer_base + activation_per_sample * per_gpu_batch
        #
        # Constants are tuned so the output lands in the documented range for each
        # model family (e.g. ResNet-50 @ batch 256 fp32 ~= 14 GB, GPT-2 124M
        # @ batch 8 fp16 ~= 12 GB), which is validated in validate_real_world.py.
        weights_optimizer_base = {
            "resnet": 2.5,
            "bert": 5.0,
            "gpt": 9.0,
            "vision_transformer": 5.5,
            "stable_diffusion": 12.0,
            "recommendation": 3.0,
        }[model_type]
        activation_per_sample = {
            "resnet": 0.045,
            "bert": 0.16,
            "gpt": 0.60,
            "vision_transformer": 0.13,
            "stable_diffusion": 1.6,
            "recommendation": 0.015,
        }[model_type]
        gpu_adjust = {"t4": 1.05, "v100": 1.0, "a100": 0.98, "rtx4090": 1.0}[gpu_type]
        per_gpu_batch = batch_size / max(num_gpus, 1)
        # Mixed precision mainly shrinks activation memory (fp16 activations),
        # while weights/optimizer state stay resident.
        activation_factor = 0.6 if mixed_precision else 1.0
        activation_mem = activation_per_sample * per_gpu_batch * activation_factor
        noise = np.random.normal(1.0, 0.05)
        return float(max(0.5, (weights_optimizer_base + activation_mem) * gpu_adjust * noise))

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
        # Memory pressure is the dominant, learnable driver of failure: jobs that
        # sit near the GPU's capacity OOM or crash far more often than jobs with
        # headroom, even before they strictly exceed it (fragmentation, peak spikes).
        utilization = peak_vram / vram_capacity
        prob = 0.02
        if utilization > 1.0:
            prob += 0.75
        elif utilization > 0.85:
            prob += 0.35
        elif utilization > 0.7:
            prob += 0.12
        if model_type in {"gpt", "stable_diffusion"} and gpu_type == "t4":
            prob += 0.10
        if batch_size >= 256:
            prob += 0.06
        if distributed_training and num_gpus >= 4:
            prob += 0.05

        failed = self.rng.random() < min(prob, 0.95)
        if not failed:
            return False, "none"

        if peak_vram > vram_capacity:
            return True, "oom"
        reason_weights = [0.2, 0.35, 0.25, 0.2]
        reason = np.random.choice(FAILURE_REASONS[:-1], p=reason_weights)
        return True, str(reason)

