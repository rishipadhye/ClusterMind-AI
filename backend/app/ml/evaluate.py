from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trained ClusterMind model artifacts.")
    parser.add_argument("--artifact-path", type=str, default="data/models/model_bundle.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found at {artifact_path}")
    bundle = joblib.load(artifact_path)
    print(json.dumps(bundle["metrics"], indent=2))
    print("Model version:", bundle["version"])
    print("Features:", len(bundle["feature_columns"]))
    if "failure_model" in bundle:
        print("Failure model type:", type(bundle["failure_model"]).__name__)
        print("Classification metrics:", json.dumps(bundle["metrics"]["failure"], indent=2))


if __name__ == "__main__":
    main()

