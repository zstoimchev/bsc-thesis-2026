# Data directory

This directory is used for local datasets required by the research experiments.

The actual dataset files are **not committed to Git** because they are large and may have separate licenses or redistribution restrictions. Only this README and `.gitkeep` placeholder files are tracked.

Expected local structure:

```text
data/
├── original_datasets/
│   ├── CIC-DDoS-2019/
│   ├── CIC-IDS-2017/
│   ├── CIC-IDS2018/
│   ├── CIC-IDS2018-normalized/
│   ├── NSL-KDD/
│   └── UNSW-NB15/
├── parquets/
│   ├── cic_superset.parquet
│   ├── realworld/
│   ├── realworld_botnet_balanced/
│   └── realworld_suricata_labeled/
├── prepared/
└── splits/
```

## Main local datasets

| Path                                        | Purpose                                           |
|---------------------------------------------|---------------------------------------------------|
| `data/original_datasets/`                   | Original datasets used by papers and repositories |
| `data/parquets/cic_superset.parquet`        | Main unified CIC-based evaluation dataset         |
| `data/parquets/realworld_botnet_balanced/`  | Trusted Danilo botnet-balanced real-world dataset |
| `data/parquets/realworld/`                  | Real-world CIC-like and NSL-like parquet datasets |
| `data/parquets/realworld_suricata_labeled/` | Real-world Suricata-labeled parquet datasets      |
| `data/prepared/`                            | Future research-prepared datasets                 |
| `data/splits/`                              | Optional shared split files                       |

## Experimental lab symlinks

The `experimental_lab` can reference this directory through local symlinks:

```bash
ln -s ../../data/original_datasets experimental_lab/data/original_datasets
ln -s ../../data/parquets experimental_lab/data/parquets
```

These symlinks are local convenience links and do not need to be committed.
