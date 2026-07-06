# Model 04: PyTorch Tabular Transformer

This model implements a custom Transformer-based binary classifier for tabular IDS/DDoS flow features.

Each input feature is treated as a token. The Transformer encoder applies attention across feature representations, and
the final pooled representation is used for binary classification.

## Purpose

The Transformer is included as the advanced neural model. It is used to compare a more expressive attention-based
architecture against XGBoost and the MLP baseline.

## Training

```bash
python run.py train-evaluate --split-id cic_random_80_20_v1 --model mdl04_transformer
```

## Notes

Training, sampling, standardization, validation, artifact saving, and prediction are handled by the shared PyTorch
trainer in `src/common/torch_training.py`.

Only the Transformer architecture is defined inside this model folder.
