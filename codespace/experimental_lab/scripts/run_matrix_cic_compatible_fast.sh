#!/usr/bin/env bash
set -u

mkdir -p results/matrix_logs

MODELS=(
  model01_deeplearning_ids_keras_tf_mlp

  model02_ddos_traffic_research_random_forest
  model02_ddos_traffic_research_xgboost
  model02_ddos_traffic_research_lightgbm
  model02_ddos_traffic_research_adaboost
  model02_ddos_traffic_research_lda
  model02_ddos_traffic_research_gaussian_nb
  model02_ddos_traffic_research_dnn

  model06_cic_ddos2019_deeplearning_gru
  model06_cic_ddos2019_deeplearning_lstm
  model06_cic_ddos2019_deeplearning_cnn_lstm

  model07_cnn_ids_1d_cnn
  model07_cnn_ids_cnn_gru
  model07_cnn_ids_cnn_lstm

  model14_dnn_ids_mlp
)

DATASETS=(
  matrix_cic_train_cic_test
  matrix_cic_train_botnet_test
  matrix_botnet_train_botnet_test
  matrix_botnet_train_cic_test
)

echo "Fast CIC-compatible matrix run"
echo "Models: ${#MODELS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Planned runs: $((${#MODELS[@]} * ${#DATASETS[@]}))"

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
