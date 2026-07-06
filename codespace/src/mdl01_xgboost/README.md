# Model 01: XGBoost

This model implements an XGBoost binary classifier for IDS/DDoS detection using tabular CIC-style flow features.

## Purpose

XGBoost is used as the strong classical machine learning baseline. It is well suited for structured tabular flow data
and provides a comparison point against neural models.

## Training

```bash
python run.py train-evaluate --split-id cic_random_80_20_v1 --model mdl01_xgboost
```

## Notes

The model trains on a balanced sample from the training split and evaluates on the full test split. It is expected to
perform strongly on tabular CIC-style features.
