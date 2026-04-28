---
title: Telco Churn API
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Telco Customer Churn API 

This repository now has two implementation steps you can skip Step 1 since it is already done:

- **Step 1 (Notebook):** train/select model and export artifacts.
- **Step 2 (API):** serve predictions through FastAPI using exported artifacts.

## Step 1: Export model artifacts from notebook

Step 1 is implemented **inside**:

- `notebooks/telco_churn_ordered_sota_modeling.ipynb`

Run notebook section `21) Phase 1 Export for API Deployment` to generate:

- `artifacts/churn_model.joblib`
- `artifacts/metadata.json`
- `artifacts/leaderboard.json`
- `artifacts/sample_payload.json`

## Step 2: Run FastAPI service

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn src.main:app --reload
```

### Endpoints

- `GET /` -> service banner
- `GET /health` -> artifact/model load status
- `POST /predict` -> single customer churn prediction
- `POST /predict-batch` -> multiple customer churn predictions

### Important

If artifacts are missing, `/predict` and `/predict-batch` return **503** with instructions.
Generate artifacts first by running notebook section 21. 
