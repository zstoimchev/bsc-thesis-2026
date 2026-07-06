# Model 02: PyTorch MLP / DNN

This model implements a feedforward neural network using PyTorch.

The architecture is a simple multilayer perceptron with fully connected layers, ReLU activations, dropout, and a binary
output logit.

## Purpose

The MLP is used as the simple neural baseline. It represents the DNN/MLP-style models commonly reported in IDS
literature.

## Training

```bash
python run.py train-evaluate --split-id cic_random_80_20_v1 --model mdl02_mlp
```

## Notes

Training, sampling, standardization, validation, artifact saving, and prediction are handled by the shared PyTorch
trainer in `src/common/torch_training.py`.

Only the model architecture is defined inside this model folder.
