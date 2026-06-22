# Data directory

This directory is reserved for datasets used during the research experiments.

The actual dataset files are **not committed to Git** because they are large and may have separate licenses or distribution restrictions.

Expected local structure:

```text
data/
├── original_datasets/
│   ├── CIC-DDoS-2019/
│   ├── CIC-IDS-2017/
│   ├── CIC-IDS2018/
│   ├── NSL-KDD/
│   └── UNSW-NB15/
└── parquets/
    └── cic_superset.parquet
```

## Important datasets

| Dataset         | Purpose                                              | Local expected path                     |
|-----------------|------------------------------------------------------|-----------------------------------------|
| CIC-DDoS2019    | Original DDoS dataset / paper reproduction reference | `data/original_datasets/CIC-DDoS-2019/` |
| CIC-IDS2017     | IDS/DDoS reference dataset                           | `data/original_datasets/CIC-IDS-2017/`  |
| CSE-CIC-IDS2018 | IDS/DDoS reference dataset                           | `data/original_datasets/CIC-IDS2018/`   |
| NSL-KDD         | Original-like reproduction for NSL-KDD-based papers  | `data/original_datasets/NSL-KDD/`       |
| UNSW-NB15       | Additional IDS reference dataset                     | `data/original_datasets/UNSW-NB15/`     |
| CIC superset    | Main unified research evaluation dataset             | `data/parquets/cic_superset.parquet`    |

## Notes

The main research comparison should use a common prepared dataset and common evaluation pipeline. The original datasets are kept locally for audit, reference, and possible original-like reproduction, but they should not be mixed directly unless their feature spaces and preprocessing are compatible.
