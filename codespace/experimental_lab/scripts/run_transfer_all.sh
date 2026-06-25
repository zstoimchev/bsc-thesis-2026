#!/usr/bin/env bash
set -u

mkdir -p results/transfer_logs

MODELS=(
  model02_ddos_traffic_research_lightgbm
  model02_ddos_traffic_research_xgboost
  model02_ddos_traffic_research_random_forest
  model14_dnn_ids_mlp
  model06_cic_ddos2019_deeplearning_gru
  model07_cnn_ids_1d_cnn
)

DATASETS=(
  transfer_cicddos2019_native_aligned
  transfer_cicddos2019_to_botnet_balanced_full_strict
)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo
    echo "===================================================================================================="
    echo "Running model=$model dataset=$dataset"
    echo "===================================================================================================="

    log="results/transfer_logs/${model}__${dataset}.log"

    /usr/bin/time -v python orchestrate.py \
      --run \
      --models "$model" \
      --dataset-id "$dataset" \
      --continue-on-error \
      2>&1 | tee "$log"

    echo "Saved log: $log"
  done
done
