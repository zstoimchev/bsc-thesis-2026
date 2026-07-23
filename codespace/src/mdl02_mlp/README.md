# Model 02: PyTorch MLP / DNN

## Overview

This model implements a multilayer perceptron using PyTorch.

Each network flow is represented as one numerical feature vector and processed independently by fully connected
neural-network layers.

## Role in the Comparison

The MLP represents a conventional feedforward neural-network baseline. It provides a comparison between XGBoost and the
more complex GRU and Transformer-style architectures.

Similar feedforward networks are commonly used in intrusion-detection research.

## Implementation

The architecture contains:

- an input layer matching the number of selected features;
- a hidden layer with 64 neurons;
- a hidden layer with 32 neurons;
- ReLU activation after each hidden layer;
- dropout of 0.1;
- one binary output logit.

Training and evaluation are handled by the shared components in:

```text
src/libraries/torch_training.py
src/libraries/torch_evaluating.py
```

The shared training procedure performs feature standardization, creates a 90/10 fitting and validation division, and
trains the model for 10 epochs using AdamW and binary cross-entropy with logits.

The complete training split is used by default. The optional `--cap` argument limits the number of training records.

## Usage

```bash
python run.py train-evaluate --model mdl02_mlp --split-id cic_random_80_20_v1
```

## Notes

The saved artifact includes the learned parameters, architecture, expected feature order, standardisation values,
training history, threshold, split identifier, and random seed.
