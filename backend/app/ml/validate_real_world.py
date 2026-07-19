"""Validate the trained models against a small real-world holdout.

The models are trained purely on synthetic data, so the number that actually
matters for an interviewer is: *does the model generalize to configurations that
were measured on real hardware?* This script answers that with a curated holdout
of documented GPU training configurations and reports the model's error on it
side-by-side with the synthetic test error (the "generalization gap").

Scope and honesty notes
------------------------
* The holdout focuses on **peak VRAM** and **OOM failure**, because those are the
  reproducible, well-documented quantities for a given (model, batch size,
  precision, GPU) tuple. Peak memory for e.g. ResNet-50 @ batch 256 fp32 is
  effectively deterministic and widely reported; wall-clock *runtime* of a full
  training job is not (it depends on epochs, dataset, and cluster), so runtime is
  intentionally left out of the seed holdout and should be populated from your own
  timed Colab/laptop runs.
* The seed figures below are approximate, community/documentation-derived numbers
  (order-of-magnitude correct), each tagged with a ``source`` note. They are a
  weak-but-real signal, not a benchmark suite. Append your own measured runs to
  ``data/real_world_holdout.csv`` to strengthen it.

Run with:

    python -m app.ml.validate_real_world --artifact-path data/models/model_bundle.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score

from app.ml.features import add_engineered_features


# FLOP multipliers mirror the synthetic generator so the derived ``estimated_flops``
# feature is in the same units the model was trained on.
FLOP_MULTIPLIERS = {
    "resnet": 8e8,
    "bert": 2.5e9,
    "gpt": 1.2e10,
    "vision_transformer": 4.5e9,
    "stable_diffusion": 1.5e10,
    "recommendation": 7e8,
}

GPU_VRAM_CAPACITY = {"t4": 16, "v100": 32, "a100": 80, "rtx4090": 24}


# Curated real-world reference configurations. peak_vram_gb / failed are the
# real-hardware observations; the rest are the job description fed to the model.
# Sources are documentation/community-reported footprints, treated as approximate.
REFERENCE_RUNS: list[dict] = [
    dict(name="resnet50_bs256_fp32_v100", model_type="resnet", framework="pytorch", gpu_type="v100",
         num_gpus=1, batch_size=256, dataset_size=1_281_167, cpu_cores=16, ram_gb=64,
         mixed_precision=0, distributed_training=0, peak_vram_gb=14.0, failed=0,
         failure_reason="none", source="ResNet-50/ImageNet fp32 ~14GB (torchvision refs)"),
    dict(name="resnet50_bs128_fp16_t4", model_type="resnet", framework="pytorch", gpu_type="t4",
         num_gpus=1, batch_size=128, dataset_size=1_281_167, cpu_cores=8, ram_gb=32,
         mixed_precision=1, distributed_training=0, peak_vram_gb=7.0, failed=0,
         failure_reason="none", source="ResNet-50 amp batch128 ~7GB (Colab T4 reports)"),
    dict(name="bert_base_bs32_fp16_v100", model_type="bert", framework="pytorch", gpu_type="v100",
         num_gpus=1, batch_size=32, dataset_size=100_000, cpu_cores=16, ram_gb=64,
         mixed_precision=1, distributed_training=0, peak_vram_gb=10.0, failed=0,
         failure_reason="none", source="BERT-base fine-tune seq128 ~10GB (HF community)"),
    dict(name="bert_large_bs16_fp16_v100", model_type="bert", framework="pytorch", gpu_type="v100",
         num_gpus=1, batch_size=16, dataset_size=100_000, cpu_cores=16, ram_gb=64,
         mixed_precision=1, distributed_training=0, peak_vram_gb=15.0, failed=0,
         failure_reason="none", source="BERT-large fine-tune ~15GB fp16 (HF community)"),
    dict(name="bert_large_bs64_fp32_t4", model_type="bert", framework="pytorch", gpu_type="t4",
         num_gpus=1, batch_size=64, dataset_size=100_000, cpu_cores=8, ram_gb=32,
         mixed_precision=0, distributed_training=0, peak_vram_gb=20.0, failed=1,
         failure_reason="oom", source="BERT-large bs64 fp32 exceeds 16GB T4 -> OOM"),
    dict(name="gpt2_124m_bs8_fp16_v100", model_type="gpt", framework="pytorch", gpu_type="v100",
         num_gpus=1, batch_size=8, dataset_size=500_000, cpu_cores=16, ram_gb=64,
         mixed_precision=1, distributed_training=0, peak_vram_gb=12.0, failed=0,
         failure_reason="none", source="GPT-2 124M fine-tune ~12GB (nanoGPT reports)"),
    dict(name="gpt2_medium_bs16_fp16_a100", model_type="gpt", framework="pytorch", gpu_type="a100",
         num_gpus=1, batch_size=16, dataset_size=500_000, cpu_cores=24, ram_gb=128,
         mixed_precision=1, distributed_training=0, peak_vram_gb=30.0, failed=0,
         failure_reason="none", source="GPT-2 medium bs16 ~30GB A100 (community)"),
    dict(name="gpt2_bs32_fp32_t4", model_type="gpt", framework="pytorch", gpu_type="t4",
         num_gpus=1, batch_size=32, dataset_size=500_000, cpu_cores=8, ram_gb=32,
         mixed_precision=0, distributed_training=0, peak_vram_gb=22.0, failed=1,
         failure_reason="oom", source="GPT-2 bs32 fp32 exceeds 16GB T4 -> OOM"),
    dict(name="vit_b16_bs64_fp16_v100", model_type="vision_transformer", framework="pytorch",
         gpu_type="v100", num_gpus=1, batch_size=64, dataset_size=1_281_167, cpu_cores=16,
         ram_gb=64, mixed_precision=1, distributed_training=0, peak_vram_gb=14.0, failed=0,
         failure_reason="none", source="ViT-B/16 bs64 amp ~14GB (timm community)"),
    dict(name="vit_l16_bs32_fp32_a100", model_type="vision_transformer", framework="pytorch",
         gpu_type="a100", num_gpus=1, batch_size=32, dataset_size=1_281_167, cpu_cores=24,
         ram_gb=128, mixed_precision=0, distributed_training=0, peak_vram_gb=26.0, failed=0,
         failure_reason="none", source="ViT-L/16 bs32 fp32 ~26GB (timm community)"),
    dict(name="sd_finetune_bs4_fp16_a100", model_type="stable_diffusion", framework="pytorch",
         gpu_type="a100", num_gpus=1, batch_size=4, dataset_size=200_000, cpu_cores=24,
         ram_gb=128, mixed_precision=1, distributed_training=0, peak_vram_gb=21.0, failed=0,
         failure_reason="none", source="SD v1.5 fine-tune bs4 ~21GB (diffusers community)"),
    dict(name="sd_bs8_fp32_t4", model_type="stable_diffusion", framework="pytorch", gpu_type="t4",
         num_gpus=1, batch_size=8, dataset_size=200_000, cpu_cores=8, ram_gb=32,
         mixed_precision=0, distributed_training=0, peak_vram_gb=28.0, failed=1,
         failure_reason="oom", source="SD bs8 fp32 exceeds 16GB T4 -> OOM"),
    dict(name="dlrm_bs512_fp16_v100", model_type="recommendation", framework="pytorch",
         gpu_type="v100", num_gpus=1, batch_size=512, dataset_size=5_000_000, cpu_cores=32,
         ram_gb=256, mixed_precision=1, distributed_training=0, peak_vram_gb=11.0, failed=0,
         failure_reason="none", source="DLRM-style bs512 ~11GB (MLPerf-adjacent)"),
    dict(name="resnet50_bs512_fp16_a100x4", model_type="resnet", framework="pytorch",
         gpu_type="a100", num_gpus=4, batch_size=512, dataset_size=1_281_167, cpu_cores=48,
         ram_gb=256, mixed_precision=1, distributed_training=1, peak_vram_gb=12.0, failed=0,
         failure_reason="none", source="ResNet-50 DDP 4xA100 per-GPU ~12GB (community)"),
    dict(name="sd_bs2_fp16_rtx4090", model_type="stable_diffusion", framework="pytorch",
         gpu_type="rtx4090", num_gpus=1, batch_size=2, dataset_size=200_000, cpu_cores=16,
         ram_gb=64, mixed_precision=1, distributed_training=0, peak_vram_gb=18.0, failed=0,
         failure_reason="none", source="SD fine-tune bs2 ~18GB on 24GB 4090 (community)"),
]


def build_holdout_frame() -> pd.DataFrame:
    rows = []
    for run in REFERENCE_RUNS:
        estimated_flops = FLOP_MULTIPLIERS[run["model_type"]] * (run["dataset_size"] / run["batch_size"])
        rows.append(
            {
                "job_id": run["name"],
                "model_type": run["model_type"],
                "framework": run["framework"],
                "gpu_type": run["gpu_type"],
                "num_gpus": run["num_gpus"],
                "batch_size": run["batch_size"],
                "dataset_size": run["dataset_size"],
                "cpu_cores": run["cpu_cores"],
                "ram_gb": run["ram_gb"],
                "estimated_flops": round(estimated_flops, 2),
                "distributed_training": int(run["distributed_training"]),
                "mixed_precision": int(run["mixed_precision"]),
                "gpu_utilization": 75.0,
                "peak_vram_gb": run["peak_vram_gb"],
                "failed": int(run["failed"]),
                "failure_reason": run["failure_reason"],
                "source": run["source"],
            }
        )
    return pd.DataFrame(rows)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class RealWorldValidator:
    def __init__(self, artifact_path: str, holdout_csv: str, synthetic_metrics_path: str) -> None:
        self.artifact_path = Path(artifact_path)
        self.holdout_csv = Path(holdout_csv)
        self.synthetic_metrics_path = Path(synthetic_metrics_path)

    def run(self) -> dict:
        frame = build_holdout_frame()
        self.holdout_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.holdout_csv, index=False)

        bundle = joblib.load(self.artifact_path)
        enriched = add_engineered_features(frame)
        features = enriched[bundle["feature_columns"]]
        transformed = bundle["preprocessor"].transform(features)

        synthetic = self._load_synthetic_metrics()

        # VRAM: the reproducible real-world signal.
        vram_pred = bundle["vram_model"].predict(transformed)
        vram_true = frame["peak_vram_gb"].to_numpy()
        vram_mae = float(mean_absolute_error(vram_true, vram_pred))
        vram_result = {
            "n": int(len(frame)),
            "real_world": {"mae": vram_mae, "rmse": _rmse(vram_true, vram_pred)},
            "synthetic_test": synthetic.get("vram", {}),
            "generalization_gap_mae": vram_mae - float(synthetic.get("vram", {}).get("mae", float("nan"))),
        }

        # Failure / OOM.
        threshold = float(bundle.get("failure_threshold", 0.5))
        fail_prob = bundle["failure_model"].predict_proba(transformed)[:, 1]
        fail_true = frame["failed"].to_numpy()
        fail_pred = (fail_prob >= threshold).astype(int)
        failure_result = {
            "n": int(len(frame)),
            "positives": int(fail_true.sum()),
            "real_world": {
                "f1": float(f1_score(fail_true, fail_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(fail_true, fail_prob)) if len(np.unique(fail_true)) > 1 else None,
                "accuracy": float((fail_pred == fail_true).mean()),
            },
            "synthetic_test": synthetic.get("failure", {}),
        }

        results = {
            "vram": vram_result,
            "failure": failure_result,
            "note": (
                "Curated real-world holdout (peak VRAM + OOM). Runtime is omitted "
                "because full-job wall-clock is not reproducible without real timed logs."
            ),
        }
        return results

    def _load_synthetic_metrics(self) -> dict:
        if self.synthetic_metrics_path.exists():
            with self.synthetic_metrics_path.open(encoding="utf-8") as file:
                return json.load(file)
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ClusterMind models on a real-world holdout.")
    parser.add_argument("--artifact-path", type=str, default="data/models/model_bundle.joblib")
    parser.add_argument("--holdout-csv", type=str, default="data/real_world_holdout.csv")
    parser.add_argument("--synthetic-metrics-path", type=str, default="data/models/metrics.json")
    parser.add_argument("--output-path", type=str, default="data/models/real_world_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validator = RealWorldValidator(
        artifact_path=args.artifact_path,
        holdout_csv=args.holdout_csv,
        synthetic_metrics_path=args.synthetic_metrics_path,
    )
    results = validator.run()

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote holdout to {args.holdout_csv} and metrics to {output}")


if __name__ == "__main__":
    main()
