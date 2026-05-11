from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from app.ml.data_pipeline import DataPipeline


class AnomalyTrainer:
    def __init__(self, dataset_path: str, artifact_path: str) -> None:
        self.dataset_path = dataset_path
        self.artifact_path = Path(artifact_path)
        self.pipeline = DataPipeline()

    def train(self) -> dict:
        frame = self.pipeline.load_dataset(self.dataset_path)
        split = self.pipeline.split(frame)
        preprocessor = self.pipeline.build_preprocessor()
        x_train_t = preprocessor.fit_transform(split.x_train)

        model = IsolationForest(
            n_estimators=250,
            contamination=0.03,
            random_state=42,
        )
        model.fit(x_train_t)

        bundle = {
            "version": "anomaly-v1",
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": split.x_train.columns.tolist(),
        }
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.artifact_path)
        return {"artifact_path": str(self.artifact_path), "version": bundle["version"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detector for ClusterMind.")
    parser.add_argument("--dataset-path", type=str, default="data/synthetic_jobs.csv")
    parser.add_argument("--artifact-path", type=str, default="data/models/anomaly_bundle.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = AnomalyTrainer(dataset_path=args.dataset_path, artifact_path=args.artifact_path).train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
