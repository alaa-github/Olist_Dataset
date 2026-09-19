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