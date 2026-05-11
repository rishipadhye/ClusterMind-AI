from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import add_engineered_features


TARGET_RUNTIME = "runtime_hours"
TARGET_VRAM = "peak_vram_gb"
TARGET_FAILURE = "failed"


@dataclass
class DatasetSplit:
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_runtime_train: pd.Series
    y_runtime_valid: pd.Series
    y_runtime_test: pd.Series
    y_vram_train: pd.Series
    y_vram_valid: pd.Series
    y_vram_test: pd.Series
    y_failure_train: pd.Series
    y_failure_valid: pd.Series
    y_failure_test: pd.Series


class DataPipeline:
    categorical_cols = ["model_type", "framework", "gpu_type"]
    numeric_cols = [
        "num_gpus",
        "batch_size",
        "dataset_size",
        "cpu_cores",
        "ram_gb",
        "estimated_flops",
        "distributed_training",
        "mixed_precision",
        "gpu_utilization",
        "flops_per_gpu",
        "memory_pressure_score",
        "compute_intensity_score",
        "gpu_efficiency_ratio",
        "oom_risk_hint",
    ]

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["failed"] = frame["failed"].astype(int)
        frame = add_engineered_features(frame)
        return frame

    def split(self, frame: pd.DataFrame) -> DatasetSplit:
        y_runtime = frame[TARGET_RUNTIME]
        y_vram = frame[TARGET_VRAM]
        y_failure = frame[TARGET_FAILURE]

        excluded = {"job_id", "timestamp", TARGET_RUNTIME, TARGET_VRAM, TARGET_FAILURE, "failure_reason"}
        feature_cols = [c for c in frame.columns if c not in excluded]
        x = frame[feature_cols].copy()

        x_train, x_temp, y_runtime_train, y_runtime_temp, y_vram_train, y_vram_temp, y_failure_train, y_failure_temp = (
            train_test_split(
                x,
                y_runtime,
                y_vram,
                y_failure,
                test_size=0.3,
                random_state=42,
                stratify=y_failure,
            )
        )
        x_valid, x_test, y_runtime_valid, y_runtime_test, y_vram_valid, y_vram_test, y_failure_valid, y_failure_test = (
            train_test_split(
                x_temp,
                y_runtime_temp,
                y_vram_temp,
                y_failure_temp,
                test_size=0.5,
                random_state=42,
                stratify=y_failure_temp,
            )
        )

        return DatasetSplit(
            x_train=x_train,
            x_valid=x_valid,
            x_test=x_test,
            y_runtime_train=y_runtime_train,
            y_runtime_valid=y_runtime_valid,
            y_runtime_test=y_runtime_test,
            y_vram_train=y_vram_train,
            y_vram_valid=y_vram_valid,
            y_vram_test=y_vram_test,
            y_failure_train=y_failure_train,
            y_failure_valid=y_failure_valid,
            y_failure_test=y_failure_test,
        )

    def build_preprocessor(self) -> Pipeline:
        transformer = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
                ("numerical", StandardScaler(), self.numeric_cols),
            ],
            remainder="drop",
        )
        return Pipeline([("transform", transformer)])

