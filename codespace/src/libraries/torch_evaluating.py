# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader
#
# from src.common.data_loader import load_dataset
# from src.common.metrics import compute_metrics
# from src.common.preprocessing import split_xy
# from src.libraries.torch_common import (
#     TorchBinaryTrainingConfig,
#     DEFAULT_TORCH_BINARY_CONFIG,
#     get_device,
#     check_numeric_features
# )
#
#
# def standardize_eval_data(
#         x_test: pd.DataFrame,
#         feature_mean: list[float],
#         feature_std: list[float],
# ) -> np.ndarray:
#     x_np = x_test.to_numpy(dtype=np.float32)
#
#     mean = np.array(feature_mean, dtype=np.float32)
#     std = np.array(feature_std, dtype=np.float32)
#
#     std[std == 0] = 1.0
#
#     x_np = (x_np - mean) / std
#     x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
#
#     return x_np.astype(np.float32)
#
#
# def evaluate_pytorch_binary_classifier(
#         model_path,
#         project_root,
#         seed: int,
#         split_id: str,
#         split_metadata: dict,
#         model_id: str,
#         model_name: str,
#         model_type: str,
#         build_model_from_artifact_fn,
#         config: TorchBinaryTrainingConfig = DEFAULT_TORCH_BINARY_CONFIG,
# ) -> dict:
#     print(f"{model_id} Evaluating {model_name}")
#     print(f"{model_id} split_id={split_id}")
#
#     device = get_device()
#     print(f"{model_id} device={device}")
#
#     artifact = torch.load(model_path, map_location=device)
#
#     if artifact.get("model_type") != model_type:
#         raise ValueError(
#             f"Loaded artifact model_type={artifact.get('model_type')} "
#             f"but expected model_type={model_type}"
#         )
#
#     expected_features = artifact["feature_columns"]
#
#     print(f"{model_id} loading test split")
#
#     test_df = load_dataset(
#         dataset_cfg={
#             "path": split_metadata["test_file"],
#             "format": "parquet",
#         },
#         project_root=project_root,
#     )
#
#     x_test, y_test = split_xy(
#         df=test_df,
#         label_column=split_metadata["label_column"],
#         feature_columns=split_metadata["feature_columns"],
#     )
#
#     missing_features = [
#         feature
#         for feature in expected_features
#         if feature not in x_test.columns
#     ]
#
#     if missing_features:
#         raise ValueError(
#             "Evaluation split is missing features expected by model: "
#             f"{missing_features}"
#         )
#
#     check_numeric_features(model_name, x_test)
#
#     x_test = x_test[expected_features]
#
#     print(f"{model_id} test shape={x_test.shape}")
#     print(f"{model_id} test label counts={y_test.value_counts().sort_index().to_dict()}")
#
#     x_np = standardize_eval_data(
#         x_test=x_test.to_frame(),
#         feature_mean=artifact["feature_mean"],
#         feature_std=artifact["feature_std"],
#     )
#
#     y_np = y_test.to_numpy(dtype=np.int64)
#
#     x_tensor = torch.tensor(x_np, dtype=torch.float32)
#
#     dataset = TensorDataset(x_tensor)
#
#     loader = DataLoader(
#         dataset,
#         batch_size=config.eval_batch_size,
#         shuffle=False,
#     )
#
#     model = build_model_from_artifact_fn(artifact).to(device)
#
#     model.load_state_dict(artifact["model_state_dict"])
#     model.eval()
#
#     threshold = artifact.get("threshold", config.threshold)
#
#     predictions = []
#
#     with torch.no_grad():
#         for (batch_x,) in loader:
#             batch_x = batch_x.to(device)
#
#             logits = model(batch_x)
#             probs = torch.sigmoid(logits)
#             preds = (probs >= threshold).long()
#
#             predictions.append(preds.cpu().numpy())
#
#     y_pred = np.concatenate(predictions)
#
#     metrics = compute_metrics(y_np, y_pred)
#
#     metrics["model_type"] = model_type
#     metrics["split_id"] = split_id
#     metrics["seed"] = seed
#     metrics["training_sample_rows"] = artifact["sampled_training_rows"]
#     metrics["training_sample_label_counts"] = artifact["sampled_label_counts"]
#     metrics["evaluation_rows"] = int(len(y_test))
#     metrics["feature_columns"] = expected_features
#     metrics["epochs"] = artifact["epochs"]
#     metrics["batch_size"] = artifact["batch_size"]
#     metrics["learning_rate"] = artifact["learning_rate"]
#     metrics["architecture"] = artifact["architecture"]
#     metrics["training_history"] = artifact["history"]
#
#     return metrics
