from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "task2" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "task2" / "val.csv"

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = MODEL_DIR / "logistic_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"

FEATURE_COLUMNS = [
    "total_price",
    "total_freight",
    "items_count",
    "total_payment",
    "max_installments",
]

TARGET_COLUMN = "is_late"


def configure_logging() -> None:
    """Configure training logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level="INFO",
    )
    logger.add(
        LOG_DIR / "training.log",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
    )


def validate_dataframe(frame: pd.DataFrame, name: str) -> None:
    """Validate the required training columns."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns) - set(frame.columns))

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {', '.join(missing_columns)}"
        )

    if frame.empty:
        raise ValueError(f"{name} is empty")

    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{name} contains missing target values")

    target_values = set(frame[TARGET_COLUMN].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{name} target must contain only 0 and 1"
        )

    if len(target_values) < 2:
        raise ValueError(
            f"{name} target must contain both classes"
        )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric features in the expected order."""
    features = frame[FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    return features.fillna(0.0)


def calculate_metrics(
    y_true,
    predictions,
    probabilities,
) -> dict[str, float]:
    """Calculate validation metrics."""
    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def train() -> dict[str, object]:
    """Train the model and register the MLflow run."""
    configure_logging()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    logger.info("Loading training data")
    train_frame = pd.read_csv(TRAIN_PATH)
    validation_frame = pd.read_csv(VALIDATION_PATH)

    validate_dataframe(train_frame, "Training data")
    validate_dataframe(validation_frame, "Validation data")

    logger.info(
        f"Training rows: {len(train_frame)}"
    )
    logger.info(
        f"Validation rows: {len(validation_frame)}"
    )

    X_train = prepare_features(train_frame)
    X_validation = prepare_features(validation_frame)

    y_train = train_frame[TARGET_COLUMN].astype("int8")
    y_validation = validation_frame[TARGET_COLUMN].astype("int8")

    logger.info("Fitting StandardScaler on training data only")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    logger.info("Training Logistic Regression model")
    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train_scaled, y_train)

    logger.info("Calculating validation metrics")
    predictions = model.predict(X_validation_scaled)
    probabilities = model.predict_proba(
        X_validation_scaled
    )[:, 1]

    metrics = calculate_metrics(
        y_true=y_validation,
        predictions=predictions,
        probabilities=probabilities,
    )

    logger.info(f"Validation metrics: {metrics}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, PREPROCESSOR_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model_name": "logistic_regression",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "metrics": metrics,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("olist_delivery_prediction")

    with mlflow.start_run() as run:
        mlflow.log_param(
            "model_name",
            "logistic_regression",
        )
        mlflow.log_param(
            "target_column",
            TARGET_COLUMN,
        )
        mlflow.log_param(
            "feature_count",
            len(FEATURE_COLUMNS),
        )
        mlflow.log_param(
            "train_rows",
            len(train_frame),
        )
        mlflow.log_param(
            "validation_rows",
            len(validation_frame),
        )
        mlflow.log_param(
            "class_weight",
            "balanced",
        )
        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(PREPROCESSOR_PATH))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(SUMMARY_PATH))

        logger.info(
            f"MLflow run completed: {run.info.run_id}"
        )

    logger.info(
        f"Model saved to: {MODEL_PATH}"
    )
    logger.info(
        f"Preprocessor saved to: {PREPROCESSOR_PATH}"
    )

    return summary


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
