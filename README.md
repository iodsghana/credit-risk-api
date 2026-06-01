# Home Credit Default Risk API

Production-style machine learning project that predicts loan default risk using the Home Credit Default Risk Kaggle competition dataset. The pipeline trains from real application records, enriches them with bureau and prior-application history, saves a champion model, and exposes scoring through FastAPI.

## Why This Project Matters

The original prototype used simulated loan data. This version uses the real Home Credit portfolio dataset:

- `application_train.csv`: 307K current loan applications with `TARGET` as the default label.
- `bureau.csv`: external credit bureau history joined on `SK_ID_CURR`.
- `previous_application.csv`: Home Credit prior application history joined on `SK_ID_CURR`.
- `HomeCredit_columns_description.csv`: data dictionary for feature interpretation.

Those joins create stronger portfolio signals such as prior refusals, active bureau accounts, total bureau debt, overdue balances, and external debt-to-credit ratio.

## Project Structure

```text
credit-risk-api/
  api/app.py                 FastAPI service
  src/preprocess.py          Home Credit joins and feature engineering
  src/train.py               Champion model training and metadata export
  src/predict.py             Model registry and scoring helpers
  data/raw/                  Local CSV drop zone, ignored by git
  models/                    Generated model artifacts, ignored by git
  monitoring/                Generated training plots/logs
  tests/test_api.py          Unit and API tests
```

## Data Setup

Download the Kaggle competition files, then place these CSVs in `data/raw/`:

```bash
kaggle competitions download -c home-credit-default-risk
```

```text
data/raw/application_train.csv
data/raw/bureau.csv
data/raw/previous_application.csv
data/raw/HomeCredit_columns_description.csv
```

The raw CSV files are intentionally gitignored because several are larger than GitHub's normal file limit.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload
```

For a faster local smoke run:

```bash
$env:TRAIN_SAMPLE_ROWS=50000
python src/train.py
```

API docs are available at `http://127.0.0.1:8000/docs`.

## Example Prediction

```bash
curl -X POST http://localhost:8000/v1/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"AMT_INCOME_TOTAL\":162000,\"AMT_CREDIT\":406597.5,\"AMT_ANNUITY\":24700.5,\"AMT_GOODS_PRICE\":351000,\"DAYS_BIRTH\":-12005,\"DAYS_EMPLOYED\":-4542,\"EXT_SOURCE_2\":0.262949,\"EXT_SOURCE_3\":0.139376}"
```

Response:

```json
{
  "request_id": "generated-uuid",
  "default_probability": 0.12,
  "risk_band": "Low",
  "model_version": "abc123def456",
  "latency_ms": 4.21
}
```

## Training Details

`src/train.py` compares logistic regression, random forest, histogram gradient boosting, and XGBoost candidates. The best held-out ROC-AUC model is saved to `models/credit_model.pkl`, with metadata written to `models/model_metadata.json`.

Feature engineering includes:

- Age, employment years, credit-to-income ratio, annuity-to-income ratio, and goods-to-credit ratio.
- Bureau account count, active and closed bureau counts, overdue days, total bureau debt, total overdue balance, and debt-to-credit ratio.
- Previous application count, approved/refused/canceled counts, total prior credit, average annuity, average down payment, and refusal rate.

## Tests

```bash
pytest tests/ -v
```

## Author

Isaac Asiedu  
Founder & CEO, Institute of Data Science (IODS) | Data Scientist & Mathematician
