from src.common.registry import load_split_registry, load_dataset_registry, load_feature_registry


def load_split_context(
        split_id: str,
        require_feature: bool = False,
) -> tuple[dict, dict, dict | None]:
    splits = load_split_registry()
    datasets = load_dataset_registry()

    if split_id not in splits:
        raise ValueError(f"Unknown split: {split_id}")

    split_cfg = splits[split_id]

    if not split_cfg.get("enabled", True):
        raise ValueError(f"Split is disabled: {split_id}")

    dataset_id = split_cfg["dataset_id"]

    if dataset_id not in datasets:
        raise ValueError(f"Unknown dataset in split '{split_id}': {dataset_id}")

    dataset_cfg = datasets[dataset_id]

    feature_cfg = None

    if require_feature:
        features = load_feature_registry()
        feature_set_id = split_cfg["feature_set_id"]

        if feature_set_id not in features:
            raise ValueError(
                f"Unknown feature set in split '{split_id}': {feature_set_id}"
            )

        feature_cfg = features[feature_set_id]

    return split_cfg, dataset_cfg, feature_cfg
