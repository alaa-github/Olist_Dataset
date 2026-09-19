from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

app = FastAPI(
    title="Olist Delivery Prediction API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    total_price: float
    total_freight: float
    items_count: int
    total_payment: float
    max_installments: int


@app.get("/")
def root() -> dict[str, str]:
    """Return basic service information."""
    return {
        "service": "olist-delivery-prediction",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Return the service health status."""
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | int]:
    """Predict whether an order will be delivered late."""
    input_data = pd.DataFrame(
        [
            {
                "total_price": request.total_price,
                "total_freight": request.total_freight,
                "items_count": request.items_count,
                "total_payment": request.total_payment,
                "max_installments": request.max_installments,
            }
        ]
    )

    transformed_data = preprocessor.transform(input_data)

    prediction = int(model.predict(transformed_data)[0])
    probability = float(model.predict_proba(transformed_data)[0, 1])

    return {
        "delivered_late": prediction,
        "probability_late": probability,
        "threshold": 0.5,
    }

