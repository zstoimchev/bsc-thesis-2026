# Model 00: Majority Class Baseline

This model is a simple majority-class baseline.

It does not learn attack patterns. During training, it counts the labels in the training split and always predicts the
most common class.

## Purpose

The baseline is used to expose the accuracy trap in imbalanced IDS/DDoS datasets. A model can achieve high accuracy by
predicting only the majority class, while still having zero attack detection capability.

## Training

```bash
python run.py train-evaluate --split-id cic_random_80_20_v1 --model mdl00_baseline
```

## Notes

This model is not intended to be competitive. It is included only as a reference point for comparison with real
classifiers.
