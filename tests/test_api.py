from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "total_price": 120.0,
            "total_freight": 20.0,
            "items_count": 2,
            "total_payment": 140.0,
            "max_installments": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "delivered_late" in body
    assert "probability_late" in body
    assert "threshold" in body
    assert 0.0 <= body["probability_late"] <= 1.0


def test_prediction_rejects_wrong_schema():
    response = client.post(
        "/predict",
        json={
            "item_count": 2,
            "order_value": 120.0,
        },
    )

    assert response.status_code == 422
