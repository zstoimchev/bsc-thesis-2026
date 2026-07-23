# Model 04: PyTorch Tabular Transformer

## Overview

This model implements a Transformer-style binary classifier for tabular network-flow data using PyTorch.

Each numerical feature is represented as a separate token. Self-attention is then applied across the feature
representations of one flow.

## Role in the Comparison

The Transformer is included as the most computationally complex neural architecture in the comparison.

It is used to examine whether attention across tabular feature representations provides an advantage over the MLP, GRU,
and XGBoost models.

## Implementation

The architecture contains:

- one value projection for each numerical feature;
- learned feature-position embeddings;
- token dimension of 32;
- two Transformer encoder layers;
- four attention heads;
- GELU activation;
- dropout of 0.1;
- mean pooling across feature tokens;
- one binary output logit.

Training and evaluation are handled by the shared components in:

```text
src/libraries/torch_training.py
src/libraries/torch_evaluating.py
```

The model uses the same standardisation, validation fraction, optimiser, batch size, epoch count, and evaluation
threshold as the other PyTorch models.

```bash
python run.py train-evaluate --model mdl04_transformer --split-id cic_random_80_20_v1
```

## Notes

The Transformer applies attention between the features of an individual flow. It does not process a temporal sequence of
multiple network flows.