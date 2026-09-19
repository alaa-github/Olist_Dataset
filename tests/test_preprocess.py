import joblib
import pandas as pd


def test_model_and_preprocessor_have_same_feature_count( ):
    model = joblib.load("models/logistic_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    assert model.n_features_in_ == preprocessor.n_features_in_
    assert model.n_features_in_ == 5


def test_preprocessor_transforms_valid_input():
    preprocessor = joblib.load("models/preprocessor.joblib")

    input_data = pd.DataFrame(
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

    transformed = preprocessor.transform(input_data)

    assert transformed.shape == (1, 5)
