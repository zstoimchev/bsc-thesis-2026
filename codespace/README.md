# **Modeling and Understanding of Modern Network Intrusion Detection Systems**

This repository contains the code for my thesis experiments on DDoS / intrusion detection using machine learning.

## Overview

Existing IDS/DDoS models often report very high accuracy, but these results are strongly affected by dataset choice, preprocessing, feature representation, class balance, and train/test splitting. Therefore, I will evaluates whether representative top-performing IDS architectures remain reliable under a unified, reproducible, deployment-oriented evaluation pipeline. In a nutshell:

## Objective

- Papers report very high accuracy results
- Those results may be inflated by dataset/split/feature issues
- KDD/NSL-like and CIC-like datasets are not naturally compatible
- I am re-implementing representative models in one clean pipeline
- Evaluate them fairly with the same preprocessing, same split, same metrics
- Test whether the models are actually suitable for firewall-style deployment

## Environment

### Conda:
```bash
conda env create -f environment.yml
conda activate thesis2026
```

### Venv:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Datasets are not committed to Git. Only placeholder files and README files are tracked.

Expected local structure:

```text
data/
├── original_datasets/
├── parquets/
├── prepared/
└── splits/
```

After restoring the local datasets, recreate the `experimental_lab` data links:

```bash
./scripts/setup_data_links.sh
```
