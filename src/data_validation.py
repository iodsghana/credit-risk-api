"""
Data validation and portfolio reporting for the Home Credit dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from preprocess import (
    APPLICATION_FILE,
    BUREAU_FILE,
    ID_COL,
    PREVIOUS_APPLICATION_FILE,
    RAW_DATA_DIR,
    TARGET_COL,
    HomeCreditDataError,
)

REQUIRED_FILES = [
    APPLICATION_FILE,
    BUREAU_FILE,
    PREVIOUS_APPLICATION_FILE,
    "HomeCredit_columns_description.csv",
]

REQUIRED_APPLICATION_COLUMNS = {
    ID_COL,
    TARGET_COL,
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
}

REQUIRED_BUREAU_COLUMNS = {
    ID_COL,
    "CREDIT_ACTIVE",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
}

REQUIRED_PREVIOUS_COLUMNS = {
    ID_COL,
    "NAME_CONTRACT_STATUS",
    "AMT_CREDIT",
    "AMT_APPLICATION",
    "DAYS_DECISION",
}


def _read_csv_with_encoding_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def validate_raw_files(raw_dir: Path = RAW_DATA_DIR) -> dict:
    """Validate file presence, schema, class balance, and join coverage."""
    missing_files = [name for name in REQUIRED_FILES if not (raw_dir / name).exists()]
    if missing_files:
        raise HomeCreditDataError(f"Missing required raw files: {missing_files}")

    app = _read_csv_with_encoding_fallback(raw_dir / APPLICATION_FILE)
    bureau = _read_csv_with_encoding_fallback(raw_dir / BUREAU_FILE)
    previous = _read_csv_with_encoding_fallback(raw_dir / PREVIOUS_APPLICATION_FILE)

    missing_columns = {
        APPLICATION_FILE: sorted(REQUIRED_APPLICATION_COLUMNS - set(app.columns)),
        BUREAU_FILE: sorted(REQUIRED_BUREAU_COLUMNS - set(bureau.columns)),
        PREVIOUS_APPLICATION_FILE: sorted(REQUIRED_PREVIOUS_COLUMNS - set(previous.columns)),
    }
    missing_columns = {name: cols for name, cols in missing_columns.items() if cols}
    if missing_columns:
        raise HomeCreditDataError(f"Missing required columns: {missing_columns}")

    app_ids = set(app[ID_COL])
    bureau_ids = set(bureau[ID_COL])
    previous_ids = set(previous[ID_COL])

    report = {
        "files": {
            APPLICATION_FILE: {"rows": int(len(app)), "columns": int(app.shape[1])},
            BUREAU_FILE: {"rows": int(len(bureau)), "columns": int(bureau.shape[1])},
            PREVIOUS_APPLICATION_FILE: {"rows": int(len(previous)), "columns": int(previous.shape[1])},
            "HomeCredit_columns_description.csv": {
                "rows": int(_read_csv_with_encoding_fallback(raw_dir / "HomeCredit_columns_description.csv").shape[0])
            },
        },
        "target": {
            "positive_class": "TARGET=1 default",
            "negative_class": "TARGET=0 repaid",
            "default_rate": round(float(app[TARGET_COL].mean()), 5),
            "class_counts": {str(k): int(v) for k, v in app[TARGET_COL].value_counts().to_dict().items()},
        },
        "join_coverage": {
            "application_ids": int(len(app_ids)),
            "with_bureau_history": int(len(app_ids & bureau_ids)),
            "with_previous_application_history": int(len(app_ids & previous_ids)),
            "bureau_coverage_rate": round(len(app_ids & bureau_ids) / len(app_ids), 5),
            "previous_application_coverage_rate": round(len(app_ids & previous_ids) / len(app_ids), 5),
        },
        "missingness_top_10": (
            app.isna().mean().sort_values(ascending=False).head(10).round(5).to_dict()
        ),
        "leakage_guardrails": [
            "TARGET is removed before model fitting.",
            "SK_ID_CURR is used only as a join key and excluded from features.",
            "Bureau and previous-application tables are aggregated to one row per applicant before joining.",
        ],
    }
    return report


def write_validation_report(raw_dir: Path = RAW_DATA_DIR, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or Path(__file__).resolve().parents[1] / "monitoring"
    output_dir.mkdir(exist_ok=True)
    report = validate_raw_files(raw_dir)
    out_path = output_dir / "data_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return out_path


if __name__ == "__main__":
    path = write_validation_report()
    print(f"Wrote validation report to {path}")
