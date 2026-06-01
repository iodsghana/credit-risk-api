"""
Home Credit risk preprocessing.

The project trains from the Home Credit Default Risk CSVs:
- application_train.csv is the modeling base table.
- bureau.csv adds external credit bureau history by SK_ID_CURR.
- previous_application.csv adds Home Credit prior application history.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"

APPLICATION_FILE = "application_train.csv"
BUREAU_FILE = "bureau.csv"
PREVIOUS_APPLICATION_FILE = "previous_application.csv"
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

APPLICATION_FEATURES = [
    ID_COL,
    TARGET_COL,
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


class HomeCreditDataError(ValueError):
    """Raised when required Home Credit data files or columns are missing."""


def _read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise HomeCreditDataError(
            f"Missing {path.name}. Place the Home Credit CSV files in {path.parent}."
        )
    return pd.read_csv(path, usecols=usecols)


def build_bureau_features(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    bureau_path = raw_dir / BUREAU_FILE
    if not bureau_path.exists():
        logger.warning("bureau.csv not found; training without bureau features.")
        return pd.DataFrame(columns=[ID_COL])

    bureau = _read_csv(bureau_path)
    grouped = bureau.groupby(ID_COL)

    features = pd.DataFrame(index=grouped.size().index)
    features["bureau_credit_count"] = grouped.size()
    features["bureau_active_count"] = grouped["CREDIT_ACTIVE"].apply(lambda s: (s == "Active").sum())
    features["bureau_closed_count"] = grouped["CREDIT_ACTIVE"].apply(lambda s: (s == "Closed").sum())
    features["bureau_days_overdue_max"] = grouped["CREDIT_DAY_OVERDUE"].max()
    features["bureau_days_overdue_sum"] = grouped["CREDIT_DAY_OVERDUE"].sum()
    features["bureau_credit_sum_total"] = grouped["AMT_CREDIT_SUM"].sum()
    features["bureau_credit_debt_total"] = grouped["AMT_CREDIT_SUM_DEBT"].sum()
    features["bureau_credit_overdue_total"] = grouped["AMT_CREDIT_SUM_OVERDUE"].sum()
    features["bureau_credit_prolong_sum"] = grouped["CNT_CREDIT_PROLONG"].sum()
    features["bureau_recent_credit_days_mean"] = grouped["DAYS_CREDIT"].mean()

    features["bureau_debt_to_credit_ratio"] = (
        features["bureau_credit_debt_total"]
        / features["bureau_credit_sum_total"].replace({0: np.nan})
    )
    return features.reset_index()


def build_previous_application_features(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    previous_path = raw_dir / PREVIOUS_APPLICATION_FILE
    if not previous_path.exists():
        logger.warning("previous_application.csv not found; training without prior-application features.")
        return pd.DataFrame(columns=[ID_COL])

    previous = _read_csv(previous_path)
    grouped = previous.groupby(ID_COL)

    features = pd.DataFrame(index=grouped.size().index)
    features["prev_application_count"] = grouped.size()
    features["prev_approved_count"] = grouped["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Approved").sum())
    features["prev_refused_count"] = grouped["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Refused").sum())
    features["prev_canceled_count"] = grouped["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Canceled").sum())
    features["prev_credit_total"] = grouped["AMT_CREDIT"].sum()
    features["prev_application_total"] = grouped["AMT_APPLICATION"].sum()
    features["prev_annuity_mean"] = grouped["AMT_ANNUITY"].mean()
    features["prev_down_payment_mean"] = grouped["AMT_DOWN_PAYMENT"].mean()
    features["prev_decision_days_mean"] = grouped["DAYS_DECISION"].mean()
    features["prev_refusal_rate"] = (
        features["prev_refused_count"] / features["prev_application_count"].replace({0: np.nan})
    )
    return features.reset_index()


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_inputs = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]
    for col in numeric_inputs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["age_years"] = (-df["DAYS_BIRTH"] / 365.25).clip(18, 100)
    df["employment_years"] = (-df["DAYS_EMPLOYED"].replace(365243, np.nan) / 365.25).clip(0, 60)
    df["credit_to_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace({0: np.nan})
    df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace({0: np.nan})
    df["goods_to_credit_ratio"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace({0: np.nan})
    return df


def load_training_data(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    app_path = raw_dir / APPLICATION_FILE
    app = _read_csv(app_path, usecols=lambda c: c in APPLICATION_FEATURES)

    missing = {ID_COL, TARGET_COL} - set(app.columns)
    if missing:
        raise HomeCreditDataError(f"{APPLICATION_FILE} is missing required columns: {sorted(missing)}")

    df = add_application_features(app)
    df = df.merge(build_bureau_features(raw_dir), on=ID_COL, how="left")
    df = df.merge(build_previous_application_features(raw_dir), on=ID_COL, how="left")
    df = df.replace([np.inf, -np.inf], np.nan)

    y = df.pop(TARGET_COL).astype(int)
    X = df.drop(columns=[ID_COL])
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, make_numeric_selector),
            ("categorical", categorical_pipe, make_categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_numeric_selector(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["number", "bool"]).columns.tolist()


def make_categorical_selector(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived Home Credit application features for inspection or notebooks."""
    return add_application_features(df)
