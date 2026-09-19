import pandas as pd
import pytest

from src.validation import validate_input_dataframe


REQUIRED_COLUMNS = [
    "total_price",
    "total_freight",
    "items_count",
    "total_payment",
    "max_installments",
]


def test_valid_dataframe_passes_validation():
    frame = pd.DataFrame(
        [
            {
                "total_price": 120.0,
                "total_freight": 20.0,
                "items_count": 2,
                "total_payment": 140.0,
                "max_installments": 2,
            }
        ]
    )

    validate_input_dataframe(
        frame=frame,
        required_columns=REQUIRED_COLUMNS,
        min_rows=1,
    )


def test_missing_columns_raise_error():
    frame = pd.DataFrame(
        [
            {
                "total_price": 120.0,
                "total_freight": 20.0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_input_dataframe(
            frame=frame,
            required_columns=REQUIRED_COLUMNS,
            min_rows=1,
        )


def test_empty_dataframe_raises_error():
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(ValueError):
        validate_input_dataframe(
            frame=frame,
            required_columns=REQUIRED_COLUMNS,
            min_rows=1,
        )
