"""Baseline comparison for the ClusterMind prediction models.

Trains deliberately simple baselines (mean predictor + a linear/logistic model on
a handful of core features) on the *same* train/test split used by the gradient
boosted models, then reports how much the tree ensembles improve over them.

The point is to prove the modeling choice mattered: if a linear regression on
"batch size + model size" were within a few percent of XGBoost/LightGBM, the extra
complexity would not be justified. Run it with:

    python -m app.ml.baseline --dataset-path data/synthetic_jobs.csv \
        --artifact-path data/models/model_bundle.joblib
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.data_pipeline import DataPipeline


# A deliberately minimal feature set: model identity + the two knobs an engineer
# would reach for first (how big is the model, how big is the batch/data).
BASELINE_CATEGORICAL = ["model_type"]
BASELINE_NUMERIC = ["batch_size", "estimated_flops", "num_gpus", "dataset_size"]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _pct_improvement(baseline: float, model: float) -> float:
    """Positive means the model reduced the error relative to the baseline."""
    if baseline == 0:
        return 0.0
    return float((baseline - model) / abs(baseline) * 100.0)


def _build_simple_preprocessor() -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), BASELINE_CATEGORICAL),
            ("numerical", StandardScaler(), BASELINE_NUMERIC),
        ],
        remainder="drop",
    )
    return Pipeline([("transform", transformer)])


class BaselineComparator:
    def __init__(self, dataset_path: str, artifact_path: str) -> None:
        self.pipeline = DataPipeline()
        self.dataset_path = dataset_path
        self.artifact_path = Path(artifact_path)

    def run(self) -> dict:
        frame = self.pipeline.load_dataset(self.dataset_path)
        split = self.pipeline.split(frame)

        simple_pre = _build_simple_preprocessor()
        x_train_s = simple_pre.fit_transform(split.x_train)
        x_test_s = simple_pre.transform(split.x_test)

        bundle = joblib.load(self.artifact_path)
        gbm_pre = bundle["preprocessor"]
        x_test_gbm = gbm_pre.transform(split.x_test[bundle["feature_columns"]])

        return {
            "runtime": self._compare_regression(
                target="runtime",
                x_train_s=x_train_s,
                x_test_s=x_test_s,
                y_train=split.y_runtime_train.to_numpy(),
                y_test=split.y_runtime_test.to_numpy(),
                gbm_model=bundle["runtime_model"],
                x_test_gbm=x_test_gbm,
                gbm_algo=bundle["metrics"]["runtime"]["algorithm"],
            ),
            "vram": self._compare_regression(
                target="vram",
                x_train_s=x_train_s,
                x_test_s=x_test_s,
                y_train=split.y_vram_train.to_numpy(),
                y_test=split.y_vram_test.to_numpy(),
                gbm_model=bundle["vram_model"],
                x_test_gbm=x_test_gbm,
                gbm_algo=bundle["metrics"]["vram"]["algorithm"],
            ),
            "failure": self._compare_classification(
                x_train_s=x_train_s,
                x_test_s=x_test_s,
                y_train=split.y_failure_train.to_numpy(),
                y_test=split.y_failure_test.to_numpy(),
                gbm_model=bundle["failure_model"],
                x_test_gbm=x_test_gbm,
                threshold=float(bundle.get("failure_threshold", 0.5)),
                gbm_algo=bundle["metrics"]["failure"]["algorithm"],
            ),
        }

    def _compare_regression(
        self,
        target: str,
        x_train_s,
        x_test_s,
        y_train: np.ndarray,
        y_test: np.ndarray,
        gbm_model,
        x_test_gbm,
        gbm_algo: str,
    ) -> dict:
        mean_model = DummyRegressor(strategy="mean").fit(x_train_s, y_train)
        linear_model = LinearRegression().fit(x_train_s, y_train)

        mean_pred = mean_model.predict(x_test_s)
        linear_pred = linear_model.predict(x_test_s)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
            gbm_pred = gbm_model.predict(x_test_gbm)

        mean_mae = float(mean_absolute_error(y_test, mean_pred))
        linear_mae = float(mean_absolute_error(y_test, linear_pred))
        gbm_mae = float(mean_absolute_error(y_test, gbm_pred))

        return {
            "metric": "mae",
            "mean_baseline": {"mae": mean_mae, "rmse": _rmse(y_test, mean_pred)},
            "linear_baseline": {"mae": linear_mae, "rmse": _rmse(y_test, linear_pred)},
            "gbm": {"mae": gbm_mae, "rmse": _rmse(y_test, gbm_pred), "algorithm": gbm_algo},
            "improvement_vs_mean_pct": _pct_improvement(mean_mae, gbm_mae),
            "improvement_vs_linear_pct": _pct_improvement(linear_mae, gbm_mae),
        }

    def _compare_classification(
        self,
        x_train_s,
        x_test_s,
        y_train: np.ndarray,
        y_test: np.ndarray,
        gbm_model,
        x_test_gbm,
        threshold: float,
        gbm_algo: str,
    ) -> dict:
        majority = DummyClassifier(strategy="most_frequent").fit(x_train_s, y_train)
        logistic = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train_s, y_train)

        maj_prob = majority.predict_proba(x_test_s)[:, 1]
        log_prob = logistic.predict_proba(x_test_s)[:, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"X does not have valid feature names")
            gbm_prob = gbm_model.predict_proba(x_test_gbm)[:, 1]

        # ROC-AUC is threshold-free; use each model's own operating point for F1.
        maj_pred = majority.predict(x_test_s)
        log_pred = (log_prob >= 0.5).astype(int)
        gbm_pred = (gbm_prob >= threshold).astype(int)

        def _auc(prob: np.ndarray) -> float:
            # A constant-probability baseline has an undefined/0.5 AUC.
            if len(np.unique(prob)) == 1:
                return 0.5
            return float(roc_auc_score(y_test, prob))

        maj_auc, log_auc, gbm_auc = _auc(maj_prob), _auc(log_prob), _auc(gbm_prob)
        maj_f1 = float(f1_score(y_test, maj_pred, zero_division=0))
        log_f1 = float(f1_score(y_test, log_pred, zero_division=0))
        gbm_f1 = float(f1_score(y_test, gbm_pred, zero_division=0))

        return {
            "metric": "roc_auc",
            "majority_baseline": {"f1": maj_f1, "roc_auc": maj_auc},
            "logistic_baseline": {"f1": log_f1, "roc_auc": log_auc},
            "gbm": {"f1": gbm_f1, "roc_auc": gbm_auc, "algorithm": gbm_algo},
            "auc_gain_vs_logistic": float(gbm_auc - log_auc),
            "f1_gain_vs_logistic": float(gbm_f1 - log_f1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GBM models against simple baselines.")
    parser.add_argument("--dataset-path", type=str, default="data/synthetic_jobs.csv")
    parser.add_argument("--artifact-path", type=str, default="data/models/model_bundle.joblib")
    parser.add_argument("--output-path", type=str, default="data/models/baseline_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparator = BaselineComparator(dataset_path=args.dataset_path, artifact_path=args.artifact_path)
    results = comparator.run()

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote baseline comparison to {output}")


if __name__ == "__main__":
    main()
