# Thesis 2026

This repository contains the code for my thesis experiments on DDoS / intrusion detection using machine learning.

## How to activate the environment

### For conda:
```bash
conda env create -f environment.yml
conda activate thesis2026
```

### For venv:
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

## Experimental lab

The audit and experiment pipeline is located in:

```text
experimental_lab/
```

To list available models, datasets, and splits:

```bash
cd experimental_lab
python orchestrate.py --list
```

To run the smoke test:

```bash
python orchestrate.py \
  --run \
  --all-enabled \
  --dataset-id _runner_core_test \
  --split-id _runner_core_split_v1 \
  --smoke-test \
  --continue-on-error
```
