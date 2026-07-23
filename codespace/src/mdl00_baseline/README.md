# Model 00: Majority-Class Baseline

## Overview

This model implements a majority-class baseline for binary network-intrusion detection.

During training, it counts the labels in the selected training split. During evaluation, every record is assigned to the
most frequent training class, independently of its feature values.

## Role in the Comparison

The baseline demonstrates how misleading accuracy can be on an imbalanced dataset. A model may achieve relatively high
accuracy by predicting only benign traffic while detecting no attacks.

It also provides a simple validation that dataset loading, split preparation, model saving, evaluation, and metric
calculation work correctly.

## Implementation

The model stores:

- the majority class;
- the training-label distribution;
- the selected feature order;
- the training split and random seed;
- the number of available and used training rows.

No feature values are used to determine the predictions.

## Usage

```bash
python run.py train-evaluate --model mdl00_baseline --split-id cic_random_80_20_v1
```

## Notes

This model is not intended to detect attacks competitively. It is included only as a reference for interpreting the
results of the learned classifiers.
