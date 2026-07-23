# Modeling and Understanding of Modern Network Intrusion Detection Systems

This repository contains the implementation and experiment artifacts for my undergraduate thesis on
machine-learning-based network intrusion and DDoS detection.

The project does not propose a new detection architecture - its purpose is to compare representative models under the
same data preparation, feature representation, split definitions, training procedure, and evaluation metrics.

## Research Objective

Intrusion-detection studies often report very high accuracy, but their results may depend strongly on the selected
dataset, preprocessing procedure, class distribution, feature set, and train-test split.

This project evaluates whether strong results remain reliable under more demanding conditions, including unseen attack
families, unseen DDoS subtypes, reduced feature sets, and external traffic from a different data distribution.

## Implemented Models

The current comparison includes:

- `mdl00_baseline` - majority-class baseline;
- `mdl01_xgboost` - XGBoost classifier;
- `mdl02_mlp` - PyTorch multilayer perceptron;
- `mdl03_gru` - PyTorch GRU classifier;
- `mdl04_transformer` - PyTorch tabular Transformer.

Each model has its own module under `src/` and uses the shared experiment pipeline.

## Datasets

Two main datasets are registered:

- `cic_collection` - the primary CIC-based dataset used for model training and controlled comparison;
- `realworld` - real server traffic combined with injected botnet records, used for external domain-shift evaluation.

The datasets are not committed to Git. They are expected at:

```text
data/raw/cic_collection.parquet
data/raw/realworld.parquet
```

Prepared train-test files are generated under:

```text
data/prepared/<split_id>/
├── train.parquet
├── test.parquet
└── metadata.json
```

## Feature Sets

The main experiments use `cic_common_v1`, which contains 11 numerical flow features shared by the CIC and external
datasets.

A larger `cic_full_v1` feature set is also available for selected CIC-only experiments.

## Experimental Configurations

Registered experiments include:

- stratified random 80/20 splitting;
- complete DDoS-family holdout;
- DDoS-LOIC-HTTP subtype holdout;
- DDoS-HOIC subtype holdout;
- full-feature LOIC-HTTP holdout;
- external evaluation on the complete real-world dataset.

The external dataset is evaluation-only. Models evaluated on it must be selected explicitly by their original training
split.

## Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate thesis2026
```

### Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Basic Usage

List the registered components:

```bash
python run.py list-datasets
python run.py list-features
python run.py list-splits
python run.py list-models
```

- Inspect a dataset: `python run.py inspect-dataset --dataset cic_collection`
- Prepare a registered split: `python run.py prepare-split --split-id cic_random_80_20_v1`
- Train and evaluate one model: `python run.py train-evaluate --model mdl01_xgboost --split-id cic_random_80_20_v1`
- Omitting `--model` runs all enabled and ready models: `python run.py train-evaluate --split-id cic_random_80_20_v1`
- The complete training split is used by default. For smaller development runs, the optional `--cap` argument limits the
  number of training records.
- Evaluate previously trained models on the external dataset:
  `python run.py evaluate --split-id realworld_external_full_v1 --trained-split-id cic_random_80_20_v1`
- Generate the current result summary: `python run.py summarize-results`

## Experiment Outputs

Each run creates a separate directory under `results/runs/`:

```text
results/runs/<timestamp>_<split_id>_<model_id>_seed<seed>/
├── model.joblib
├── training_summary.json
├── metrics.json
└── run_info.json
```

Depending on the command, some files may not be created. For example, evaluation-only runs reuse an existing model
artifact instead of saving a new model.

The saved metadata records the model, dataset, feature set, training and evaluation splits, random seed, execution time,
class distributions, metrics, and confusion matrix.

## Evaluation Metrics

The comparison records:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- macro and weighted F1-score;
- confusion matrix;
- training, evaluation, and total execution time.

Balanced accuracy, recall, F1-score, and the confusion matrix are especially important because ordinary accuracy can be
misleading on imbalanced datasets.

# Contact

[zstoimchev@gmail.com](mailto:zstoimchev@gmail.com)