# Olist Delivery Prediction

This repository contains the complete Olist MLOps project.

## Task 1

Task 1 loads the Olist CSV files into PostgreSQL using Docker, SQL schema definitions, COPY commands, row-count checks, and join validation.

## Task 2

Task 2 contains notebooks for data joining, label creation, data splitting, exploratory analysis, feature engineering, and Logistic Regression modeling.

## Task 3

Task 3 provides a production-style inference service with:

- Reusable preprocessing
- Input validation
- Logging
- MLflow experiment tracking
- FastAPI inference API
- Automated tests
- Docker packaging

## Install dependencies

```bash
python -m pip install -r requirements.txt

## Train the model

```bash
python -m src.train

##Run tests

```bash
python -m pytest -q

##Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

##Health check

```bash
curl http://localhost:8000/health
P
##rediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_price": 120.0,
    "total_freight": 20.0,
    "items_count": 2,
    "total_payment": 140.0,
    "max_installments": 2
  }'

##Run MLflow

```bash
python -m mlflow ui \
  --backend-store-uri ./mlruns \
  --host 0.0.0.0 \
  --port 5000

##Build Docker image

```bash
docker build -t olist-delivery-api .

##Run Docker container

```bash
docker run --rm -p 8001:8000 olist-delivery-api