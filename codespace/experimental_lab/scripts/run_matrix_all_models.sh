#!/usr/bin/env bash
set -u

mkdir -p results/matrix_logs

mapfile -t MODELS < <(
python - <<'PY'
import yaml
from pathlib import Path

data = yaml.safe_load(Path("registries/models.yaml").read_text())
models = data.get("models", data if isinstance(data, list) else [])

for m in models:
    if not isinstance(m, dict):
        continue
    if m.get("enabled", True) is False:
        continue
    if m.get("smoke_test", False) is True:
        continue

    mid = m.get("id")
    if mid:
        print(mid)
PY
)

DATASETS=(
  matrix_cic_train_cic_test
  matrix_cic_train_botnet_test
  matrix_botnet_train_botnet_test
  matrix_botnet_train_cic_test
)

echo "Models found: ${#MODELS[@]}"
printf ' - %s\n' "${MODELS[@]}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    log="results/matrix_logs/${model}__${dataset}.log"

    if [[ -f "$log" ]] && grep -q "\[done\]\|\[summary\]" "$log"; then
      echo "Skipping existing: $log"
      continue
    fi

    echo
    echo "===================================================================================================="
    echo "Running model=$model dataset=$dataset"
    echo "===================================================================================================="

    /usr/bin/time -v python orchestrate.py \
      --run \
      --models "$model" \
      --dataset-id "$dataset" \
      --continue-on-error \
      2>&1 | tee "$log"
  done
done
