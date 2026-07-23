# Model 03: PyTorch GRU

## Overview

This model implements a gated recurrent unit classifier using PyTorch.

Each network flow is represented by one numerical feature vector. For this implementation, the features of that flow are
treated as sequence positions and passed individually to the GRU.

## Role in the Comparison

The GRU represents recurrent neural-network approaches reported in previous intrusion- and DDoS-detection research.

It is compared with XGBoost, the MLP, and the Transformer-style model under the same data preparation and evaluation
conditions.

## Implementation

The architecture contains:

- one unidirectional GRU layer;
- input size of 1;
- hidden size of 64;
- one recurrent layer;
- dropout of 0.1 in the final classifier;
- layer normalisation;
- one binary output logit.

The last hidden state produced by the GRU is used for classification.

Training and evaluation are handled by the shared components in:

```text
src/libraries/torch_training.py
src/libraries/torch_evaluating.py
```

The final configuration uses:

- 10 training epochs;
- AdamW optimisation;
- binary cross-entropy with logits;
- fixed prediction threshold of 0.5;
- no class weighting;
- no validation-based threshold tuning.

## Usage

```bash
python run.py train-evaluate --model mdl03_gru --split-id cic_random_80_20_v1
```

## Notes

The model does not process consecutive network flows over time. It treats the features of one flow as an ordered
sequence. Its results must therefore not be interpreted as evidence that it learns temporal behaviour between network
communications.
