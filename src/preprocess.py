from __future__ import annotations

from typing import Iterable

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "total_price",
    "total_freight",
    "items_count",
    "total_payment",
    "max_installments",
]


def validate_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> None:
    """Validate that all required model features are available."""
    feature_columns = list(feature_columns)
    missing_columns = sorted(
        set(feature_columns) - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing feature columns: "
            + ", ".join(missing_columns)
        )


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Prepare numeric features in the expected order."""
    feature_columns = list(feature_columns)
    validate_feature_columns(frame, feature_columns)

    features = frame[feature_columns].copy()

    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    return features.fillna(0.0)


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit the scaler on training data only."""
    features = prepare_features(frame, feature_columns)

    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)

    transformed_frame = pd.DataFrame(
        transformed,
        columns=list(feature_columns),
        index=frame.index,
    )

    return scaler, transformed_frame


def transform_features(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Transform new data using a fitted scaler."""
    features = prepare_features(frame, feature_columns)
    transformed = scaler.transform(features)

    return pd.DataFrame(
        transformed,
        columns=list(feature_columns),
        index=frame.index,
    )


def save_preprocessor(
    scaler: StandardScaler,
    path: str,
) -> None:
    """Save the fitted preprocessing object."""
    joblib.dump(scaler, path)


def load_preprocessor(path: str) -> StandardScaler:
    """Load a saved preprocessing object."""
    return joblib.load(path)
