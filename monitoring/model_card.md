# Model Card: Home Credit Default Risk

## Intended Use

Estimate loan default probability for Home Credit application records. This is a portfolio-grade risk scoring system, not a production credit decision engine.

## Dataset

- Source: Home Credit Default Risk Kaggle competition.
- Rows used for this run: 20,000
- Holdout split: 4,000 rows
- Observed default rate: 8.23%
- Run mode: sampled smoke run for fast portfolio validation. Run `python src/train.py` without `TRAIN_SAMPLE_ROWS` for a full-dataset model card.

## Model Selection

Champion model: `xgboost`

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.733 |
| Average precision | 0.2274 |
| Brier score | 0.1606 |

Confusion matrix at threshold 0.50:

```text
[[TN=2864, FP=807],
 [FN=145, TP=184]]
```

## Top Permutation Importances

- `EXT_SOURCE_2`: 0.056765
- `EXT_SOURCE_3`: 0.031127
- `EXT_SOURCE_1`: 0.022927
- `prev_credit_total`: 0.007001
- `goods_to_credit_ratio`: 0.00647
- `DAYS_BIRTH`: 0.006363
- `prev_decision_days_mean`: 0.005244
- `CODE_GENDER`: 0.004363
- `bureau_debt_to_credit_ratio`: 0.003669
- `credit_to_income_ratio`: 0.003343

## Risk and Governance Notes

- `TARGET` is removed before training.
- `SK_ID_CURR` is used only for joins and removed before fitting.
- The API returns a probability and risk band; final lending decisions require policy, fairness, compliance, and human governance.
- Raw Kaggle CSVs are not committed because of size and license considerations.
