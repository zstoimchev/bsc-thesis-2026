# Model runners

This directory contains runner scripts for the model variants used in the experimental lab.

Each runner provides a common command-line interface so that different models can be evaluated using the same dataset loading, preprocessing, train/test split, and metrics pipeline.

The runners are clean local implementations or adapters based on the architectures, methods, and repository structure of the original papers. They are not intended to redistribute or directly copy the original repositories. Original repositories are credited in:

```text
experimental_lab/registries/external_repos.tsv
experimental_lab/external/README.md
```

AI assistance was used during development for debugging, refactoring, and code organization. The final code was manually reviewed and tested using the experimental lab smoke-test pipeline.

Generated models, logs, metrics, and experiment outputs are not stored in this directory. They are written under:

```text
experimental_lab/results/
```
