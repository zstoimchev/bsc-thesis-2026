# Model 01: XGBoost

## Overview

This model implements an XGBoost binary classifier for detecting malicious network flows represented by numerical
CIC-style features.

XGBoost constructs an ensemble of decision trees sequentially, with each tree attempting to correct errors made by the
previous trees.

## Role in the Comparison

XGBoost represents classical machine learning for structured tabular data. It provides a comparison with the MLP, GRU,
and Transformer-style neural models.

Tree-based ensemble methods have achieved strong results in previous DDoS-detection research and are generally well
suited to flow-level numerical features.

## Implementation

The classifier uses:

- 200 trees;
- maximum tree depth of 6;
- learning rate of 0.1;
- row subsampling of 0.8;
- feature subsampling of 0.8;
- binary logistic objective;
- histogram-based tree construction.

The model uses numerical values directly and does not require feature standardisation.

The complete prepared training split is used by default. The optional `--cap` argument can limit the number of training
records while preserving approximately the original class distribution.

## Usage

```bash
python run.py train-evaluate --model mdl01_xgboost --split-id cic_random_80_20_v1
```

## Notes

The trained XGBoost model, feature order, training metadata, parameters, and class distribution are stored together in
the experiment artifact.
