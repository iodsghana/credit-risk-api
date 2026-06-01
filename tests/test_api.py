import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from data_validation import validate_raw_files, write_validation_report
from preprocess import (
    HomeCreditDataError,
    add_application_features,
    build_bureau_features,
    build_previous_application_features,
    load_training_data,
)
from predict import align_to_training_features


def home_credit_payload() -> dict:
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 162000,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12005,
        "DAYS_EMPLOYED": -4542,
        "EXT_SOURCE_2": 0.262949,
        "EXT_SOURCE_3": 0.139376,
    }


def test_application_ratios_are_added():
    df = pd.DataFrame([home_credit_payload()])
    result = add_application_features(df)
    assert "age_years" in result.columns
    assert "credit_to_income_ratio" in result.columns
    assert result["age_years"].iloc[0] > 18


def test_bureau_features_aggregate_by_customer(tmp_path):
    raw = tmp_path
    pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "CREDIT_ACTIVE": "Active",
                "CREDIT_DAY_OVERDUE": 10,
                "AMT_CREDIT_SUM": 1000,
                "AMT_CREDIT_SUM_DEBT": 500,
                "AMT_CREDIT_SUM_OVERDUE": 25,
                "CNT_CREDIT_PROLONG": 1,
                "DAYS_CREDIT": -100,
            },
            {
                "SK_ID_CURR": 1,
                "CREDIT_ACTIVE": "Closed",
                "CREDIT_DAY_OVERDUE": 0,
                "AMT_CREDIT_SUM": 2000,
                "AMT_CREDIT_SUM_DEBT": 0,
                "AMT_CREDIT_SUM_OVERDUE": 0,
                "CNT_CREDIT_PROLONG": 0,
                "DAYS_CREDIT": -500,
            },
        ]
    ).to_csv(raw / "bureau.csv", index=False)

    result = build_bureau_features(raw)
    assert result.loc[0, "bureau_credit_count"] == 2
    assert result.loc[0, "bureau_active_count"] == 1
    assert result.loc[0, "bureau_days_overdue_sum"] == 10


def test_previous_application_features_aggregate_by_customer(tmp_path):
    raw = tmp_path
    pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "NAME_CONTRACT_STATUS": "Approved",
                "AMT_CREDIT": 1000,
                "AMT_APPLICATION": 900,
                "AMT_ANNUITY": 50,
                "AMT_DOWN_PAYMENT": 20,
                "DAYS_DECISION": -100,
            },
            {
                "SK_ID_CURR": 1,
                "NAME_CONTRACT_STATUS": "Refused",
                "AMT_CREDIT": 500,
                "AMT_APPLICATION": 700,
                "AMT_ANNUITY": 40,
                "AMT_DOWN_PAYMENT": 0,
                "DAYS_DECISION": -200,
            },
        ]
    ).to_csv(raw / "previous_application.csv", index=False)

    result = build_previous_application_features(raw)
    assert result.loc[0, "prev_application_count"] == 2
    assert result.loc[0, "prev_refused_count"] == 1
    assert result.loc[0, "prev_refusal_rate"] == 0.5


def test_prediction_payload_aligns_to_training_features():
    df = pd.DataFrame([{"AMT_CREDIT": 1000}])
    aligned = align_to_training_features(df, {"features": ["AMT_CREDIT", "bureau_credit_count"]})
    assert aligned.columns.tolist() == ["AMT_CREDIT", "bureau_credit_count"]
    assert pd.isna(aligned["bureau_credit_count"].iloc[0])


def write_minimal_raw_dataset(raw: Path):
    pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "TARGET": 0,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "F",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 162000,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -12005,
                "DAYS_EMPLOYED": -4542,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 2,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "EXT_SOURCE_1": 0.1,
                "EXT_SOURCE_2": 0.262949,
                "EXT_SOURCE_3": 0.139376,
                "OBS_30_CNT_SOCIAL_CIRCLE": 0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
                "AMT_REQ_CREDIT_BUREAU_MON": 0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1,
            },
            {
                "SK_ID_CURR": 2,
                "TARGET": 1,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "N",
                "CNT_CHILDREN": 1,
                "AMT_INCOME_TOTAL": 90000,
                "AMT_CREDIT": 300000,
                "AMT_ANNUITY": 20000,
                "AMT_GOODS_PRICE": 280000,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -9000,
                "DAYS_EMPLOYED": -100,
                "OCCUPATION_TYPE": "Sales staff",
                "CNT_FAM_MEMBERS": 2,
                "REGION_RATING_CLIENT": 3,
                "REGION_RATING_CLIENT_W_CITY": 3,
                "EXT_SOURCE_1": 0.2,
                "EXT_SOURCE_2": 0.3,
                "EXT_SOURCE_3": 0.4,
                "OBS_30_CNT_SOCIAL_CIRCLE": 1,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 1,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
                "AMT_REQ_CREDIT_BUREAU_MON": 1,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 2,
            },
        ]
    ).to_csv(raw / "application_train.csv", index=False)
    pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "CREDIT_ACTIVE": "Active",
                "CREDIT_DAY_OVERDUE": 0,
                "AMT_CREDIT_SUM": 1000,
                "AMT_CREDIT_SUM_DEBT": 100,
                "AMT_CREDIT_SUM_OVERDUE": 0,
                "CNT_CREDIT_PROLONG": 0,
                "DAYS_CREDIT": -300,
            }
        ]
    ).to_csv(raw / "bureau.csv", index=False)
    pd.DataFrame(
        [
            {
                "SK_ID_CURR": 2,
                "NAME_CONTRACT_STATUS": "Refused",
                "AMT_CREDIT": 500,
                "AMT_APPLICATION": 700,
                "AMT_ANNUITY": 40,
                "AMT_DOWN_PAYMENT": 0,
                "DAYS_DECISION": -200,
            }
        ]
    ).to_csv(raw / "previous_application.csv", index=False)
    pd.DataFrame(
        [{"Table": "application_{train|test}.csv", "Row": "TARGET", "Description": "Target"}]
    ).to_csv(raw / "HomeCredit_columns_description.csv", index=False)


def test_load_training_data_joins_and_removes_keys(tmp_path):
    write_minimal_raw_dataset(tmp_path)
    X, y = load_training_data(tmp_path)
    assert len(X) == 2
    assert y.tolist() == [0, 1]
    assert "SK_ID_CURR" not in X.columns
    assert "TARGET" not in X.columns
    assert "bureau_credit_count" in X.columns
    assert "prev_refusal_rate" in X.columns


def test_data_validation_report_covers_schema_and_joins(tmp_path):
    write_minimal_raw_dataset(tmp_path)
    report = validate_raw_files(tmp_path)
    assert report["files"]["application_train.csv"]["rows"] == 2
    assert report["target"]["default_rate"] == 0.5
    assert report["join_coverage"]["with_bureau_history"] == 1

    out_path = write_validation_report(tmp_path, tmp_path)
    assert out_path.exists()


def test_data_validation_fails_when_required_file_missing(tmp_path):
    try:
        validate_raw_files(tmp_path)
    except HomeCreditDataError as exc:
        assert "Missing required raw files" in str(exc)
    else:
        raise AssertionError("Expected HomeCreditDataError")


class TestAPI:
    def setup_method(self):
        from fastapi.testclient import TestClient
        from app import app

        mock_result = {
            "default_probability": 0.12,
            "risk_band": "Low",
            "model_version": "abc123",
        }
        self.patches = [
            patch("app.predict", return_value=mock_result),
            patch("app.predict_batch", return_value=[mock_result]),
            patch("app.ModelRegistry.load", return_value=(MagicMock(), {"sha256": "abc123def456"})),
        ]
        for active_patch in self.patches:
            active_patch.start()
        self.client = TestClient(app)

    def teardown_method(self):
        for active_patch in self.patches:
            active_patch.stop()

    def test_root_returns_200(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "Home Credit" in response.json()["service"]

    def test_predict_valid_application(self):
        response = self.client.post("/v1/predict", json=home_credit_payload())
        assert response.status_code == 200
        assert response.json()["default_probability"] == 0.12

    def test_predict_rejects_invalid_birth_days(self):
        payload = home_credit_payload()
        payload["DAYS_BIRTH"] = 1
        response = self.client.post("/v1/predict", json=payload)
        assert response.status_code == 422

    def test_batch_predict(self):
        response = self.client.post("/v1/predict/batch", json={"applications": [home_credit_payload()]})
        assert response.status_code == 200
        assert response.json()["count"] == 1

    def test_latency_header_present(self):
        response = self.client.post("/v1/predict", json=home_credit_payload())
        assert "X-Response-Time-Ms" in response.headers
