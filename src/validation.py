from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_input_dataframe(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    min_rows: int = 1,
) -> None:
    """Validate the input DataFrame schema and basic data quality."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if len(frame) < min_rows:
        raise ValueError(
            f"Input must contain at least {min_rows} row or rows"
        )

    required_columns = list(required_columns)

    missing_columns = sorted(
        set(required_columns) - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(missing_columns)
        )

    if frame[required_columns].isna().all(axis=None):
        raise ValueError(
            "All required input values are missing"
        )

    if (
        "order_purchase_timestamp" in frame.columns
        and frame["order_purchase_timestamp"].isna().all()
    ):
        raise ValueError(
            "order_purchase_timestamp cannot be entirely missing"
        )


def validate_training_dataframe(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str,
) -> None:
    """Validate the training dataset before model fitting."""
    required_columns = list(feature_columns) + [target_column]

    validate_input_dataframe(
        frame=frame,
        required_columns=required_columns,
        min_rows=2,
    )

    if frame[target_column].isna().any():
        raise ValueError(
            "Training target contains missing values"
        )

    unique_targets = set(
        frame[target_column].dropna().unique()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "Training target must contain only 0 and 1"
        )

    if len(unique_targets) < 2:
        raise ValueError(
            "Training target must contain both classes"
        )
