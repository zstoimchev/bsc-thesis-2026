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
