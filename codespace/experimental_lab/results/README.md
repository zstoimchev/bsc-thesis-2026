# Results directory

This directory is reserved for generated experiment outputs.

Generated results are **not committed to Git** by default. They can be recreated by running the orchestrator.

Typical generated structure:

```text
results/
├── artifacts/
├── logs/
├── reports/
├── runs/
├── splits/
└── runs_index.csv
```

## Contents

| Path                     | Description                                                            |
|--------------------------|------------------------------------------------------------------------|
| `results/artifacts/`     | Trained models, preprocessors, encoders, and other generated artifacts |
| `results/logs/`          | Terminal logs from experiment runs                                     |
| `results/reports/`       | Small generated summaries and exported reports                         |
| `results/runs/`          | Per-run metrics, confusion matrices, and metadata                      |
| `results/splits/`        | Saved train/test split indices                                         |
| `results/runs_index.csv` | Compact index of all executed runs                                     |

## Reproducibility note

Important final results can be exported later into a small research report table or summary file. Large generated artifacts should remain outside Git unless there is a specific reason to preserve them.
