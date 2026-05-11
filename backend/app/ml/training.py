from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score

from app.ml.data_pipeline import DataPipeline
from app.ml.experiment_tracker import ExperimentRecord, JsonExperimentTracker

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    # Coarse grid is good enough for an MVP and keeps training fast.
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        score = float(f1_score(y_true, pred))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


class ModelTrainer:
    def __init__(self, dataset_path: str, artifacts_dir: str, experiments_path: str) -> None:
        self.pipeline = DataPipeline()
        self.dataset_path = dataset_path
        self.artifacts_dir = Path(artifacts_dir)
        self.tracker = JsonExperimentTracker(experiments_path)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        frame = self.pipeline.load_dataset(self.dataset_path)
        split = self.pipeline.split(frame)
        preprocessor = self.pipeline.build_preprocessor()

        x_train_t = preprocessor.fit_transform(split.x_train)
        x_valid_t = preprocessor.transform(split.x_valid)
        x_test_t = preprocessor.transform(split.x_test)

        # Stats used later for confidence calibration.
        target_stats = {
            "runtime": {"std": float(np.std(split.y_runtime_train.to_numpy()))},
            "vram": {"std": float(np.std(split.y_vram_train.to_numpy()))},
        }

        runtime_pack = self._train_regressor(
            x_train=x_train_t,
            y_train=split.y_runtime_train.to_numpy(),
            x_valid=x_valid_t,
            y_valid=split.y_runtime_valid.to_numpy(),
            x_test=x_test_t,
            y_test=split.y_runtime_test.to_numpy(),
        )

        vram_pack = self._train_regressor(
            x_train=x_train_t,
            y_train=split.y_vram_train.to_numpy(),
            x_valid=x_valid_t,
            y_valid=split.y_vram_valid.to_numpy(),
            x_test=x_test_t,
            y_test=split.y_vram_test.to_numpy(),
        )

        failure_pack = self._train_classifier(
            x_train=x_train_t,
            y_train=split.y_failure_train.to_numpy(),
            x_valid=x_valid_t,
            y_valid=split.y_failure_valid.to_numpy(),
            x_test=x_test_t,
            y_test=split.y_failure_test.to_numpy(),
        )

        run_id = uuid4().hex[:12]
        bundle = {
            "run_id": run_id,
            "version": f"v2-{run_id}",
            "preprocessor": preprocessor,
            "runtime_model": runtime_pack["model"],
            "vram_model": vram_pack["model"],
            "failure_model": failure_pack["model"],
            "failure_threshold": failure_pack["threshold"],
            "metrics": {
                "runtime": runtime_pack["metrics"],
                "vram": vram_pack["metrics"],
                "failure": failure_pack["metrics"],
            },
            "target_stats": target_stats,
            "feature_columns": split.x_train.columns.tolist(),
        }

        artifact = self.artifacts_dir / "model_bundle.joblib"
        joblib.dump(bundle, artifact)
        self._log_records(run_id=run_id, artifact_path=str(artifact), results=bundle["metrics"])
        with (self.artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(bundle["metrics"], file, indent=2)
        return {"artifact_path": str(artifact), "metrics": bundle["metrics"], "version": bundle["version"]}

    def _train_regressor(
        self,
        x_train,
        y_train: np.ndarray,
        x_valid,
        y_valid: np.ndarray,
        x_test,
        y_test: np.ndarray,
    ) -> dict:
        candidates: list[tuple[str, object]] = [
            ("random_forest", RandomForestRegressor(n_estimators=260, random_state=42))
        ]
        if XGBRegressor:
            candidates.append(
                (
                    "xgboost",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=8,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                    ),
                )
            )
        if LGBMRegressor:
            candidates.append(
                (
                    "lightgbm",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=64,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                    ),
                )
            )

        best_model = None
        best_algo = ""
        best_score = float("inf")

        for algo, model in candidates:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
                model.fit(x_train, y_train)
                pred = model.predict(x_valid)
            score = rmse(y_valid, pred)
            if score < best_score:
                best_model = model
                best_algo = algo
                best_score = score

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
            test_pred = best_model.predict(x_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "rmse": rmse(y_test, test_pred),
            "algorithm": best_algo,
        }
        return {"model": best_model, "metrics": metrics}

    def _train_classifier(
        self,
        x_train,
        y_train: np.ndarray,
        x_valid,
        y_valid: np.ndarray,
        x_test,
        y_test: np.ndarray,
    ) -> dict:
        candidates: list[tuple[str, object]] = [
            ("random_forest", RandomForestClassifier(n_estimators=420, random_state=42, class_weight="balanced"))
        ]
        if XGBClassifier:
            candidates.append(
                (
                    "xgboost",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=7,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                )
            )
        if LGBMClassifier:
            candidates.append(
                (
                    "lightgbm",
                    LGBMClassifier(
                        n_estimators=650,
                        learning_rate=0.04,
                        num_leaves=64,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                    ),
                )
            )

        best_model = None
        best_algo = ""
        best_auc = -1.0

        for algo, model in candidates:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
                model.fit(x_train, y_train)
                prob = model.predict_proba(x_valid)[:, 1]
            auc = float(roc_auc_score(y_valid, prob))
            if auc > best_auc:
                best_model = model
                best_algo = algo
                best_auc = auc

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
            test_prob = best_model.predict_proba(x_test)[:, 1]

        # Tune decision threshold on validation (improves F1 over fixed 0.5).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
            valid_prob = best_model.predict_proba(x_valid)[:, 1]
        threshold, valid_f1 = _best_f1_threshold(y_valid, valid_prob)

        test_pred = (test_prob >= threshold).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, test_pred)),
            "roc_auc": float(roc_auc_score(y_test, test_prob)),
            "valid_f1": float(valid_f1),
            "threshold": float(threshold),
            "algorithm": best_algo,
        }
        return {"model": best_model, "metrics": metrics, "threshold": float(threshold)}

    def _log_records(self, run_id: str, artifact_path: str, results: dict) -> None:
        for target, metrics in results.items():
            self.tracker.log(
                ExperimentRecord(
                    run_id=run_id,
                    started_at=self.tracker.now_iso(),
                    model_family="tree_ensemble",
                    target=target,
                    algorithm=metrics["algorithm"],
                    metrics={k: float(v) for k, v in metrics.items() if k != "algorithm"},
                    params={"seed": 42},
                    artifact_path=artifact_path,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ClusterMind model bundle.")
    parser.add_argument("--dataset-path", type=str, default="data/synthetic_jobs.csv")
    parser.add_argument("--artifacts-dir", type=str, default="data/models")
    parser.add_argument("--experiments-path", type=str, default="data/experiments/runs.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = ModelTrainer(
        dataset_path=args.dataset_path,
        artifacts_dir=args.artifacts_dir,
        experiments_path=args.experiments_path,
    )
    result = trainer.train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
