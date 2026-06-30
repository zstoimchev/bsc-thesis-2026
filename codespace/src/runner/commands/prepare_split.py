from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.data_loader import load_dataset
from src.common.label_mapping import normalize_binary_labels
from src.common.metrics import save_json
from src.common.preprocessing import clean_dataframe_columns, clean_column_name
from src.runner.paths import PROJECT_ROOT
from src.runner.registry import load_dataset_registry, load_feature_registry, load_split_registry


def add_prepare_split_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "prepare-split",
        help="Prepare train/test files from a registered split.",
    )

    parser.add_argument(
        "--split-id",
        required=True,
        help="Split ID from registries/splits.json.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared split files.",
    )


def run_prepare_split(args) -> None:
    datasets = load_dataset_registry()
    features = load_feature_registry()
    splits = load_split_registry()

    split_id = args.split_id

    if split_id not in splits:
        raise ValueError(f"Unknown split: {split_id}")

    split_cfg = splits[split_id]

    if not split_cfg.get("enabled", True):
        raise ValueError(f"Split is disabled: {split_id}")

    dataset_id = split_cfg["dataset_id"]
    feature_set_id = split_cfg["feature_set_id"]

    if dataset_id not in datasets:
        raise ValueError(f"Unknown dataset in split '{split_id}': {dataset_id}")

    if feature_set_id not in features:
        raise ValueError(f"Unknown feature set in split '{split_id}': {feature_set_id}")

    dataset_cfg = datasets[dataset_id]
    feature_cfg = features[feature_set_id]

    output_cfg = split_cfg["output"]

    prepared_dir = PROJECT_ROOT / output_cfg["output_dir"]

    prepared_root = (PROJECT_ROOT / "data" / "prepared").resolve()
    prepared_dir = prepared_dir.resolve()

    if prepared_root not in prepared_dir.parents and prepared_dir != prepared_root:
        raise ValueError(
            f"Invalid output_dir for split '{split_id}': {prepared_dir}\n"
            f"Prepared split output must be under: {prepared_root}"
        )

    train_path = prepared_dir / output_cfg["train_file"]
    test_path = prepared_dir / output_cfg["test_file"]
    metadata_path = prepared_dir / output_cfg["metadata_file"]

    if prepared_dir.exists() and not args.overwrite:
        if train_path.exists() or test_path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"Prepared split already exists: {prepared_dir}. "
                "Use --overwrite to regenerate it."
            )

    prepared_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare-split] split_id={split_id}")
    print(f"[prepare-split] dataset_id={dataset_id}")
    print(f"[prepare-split] feature_set_id={feature_set_id}")

    df = load_dataset(dataset_cfg=dataset_cfg, project_root=PROJECT_ROOT)
    df = clean_dataframe_columns(df)

    label_column = clean_column_name(split_cfg.get("label_column") or dataset_cfg["label_column"])

    if label_column not in df.columns:
        raise ValueError(f"Label column not found after cleaning: {label_column}")

    df[label_column] = normalize_binary_labels(df[label_column])

    train_df, test_df = create_random_split(
        df=df,
        split_cfg=split_cfg,
        label_column=label_column,
    )

    train_df, final_features, dropped_columns = select_final_columns(
        df=train_df,
        dataset_cfg=dataset_cfg,
        split_cfg=split_cfg,
        feature_cfg=feature_cfg,
        label_column=label_column,
    )

    test_df, test_features, _ = select_final_columns(
        df=test_df,
        dataset_cfg=dataset_cfg,
        split_cfg=split_cfg,
        feature_cfg=feature_cfg,
        label_column=label_column,
    )

    if final_features != test_features:
        raise ValueError("Train and test feature columns do not match.")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "split_id": split_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "dataset_name": dataset_cfg.get("name", ""),
        "dataset_path": dataset_cfg.get("path"),
        "feature_set_id": feature_set_id,
        "prepared_dir": output_cfg["output_dir"],
        "train_file": str(train_path.relative_to(PROJECT_ROOT)),
        "test_file": str(test_path.relative_to(PROJECT_ROOT)),
        "label_column": label_column,
        "split_method": split_cfg["split_method"],
        "feature_columns": final_features,
        "dropped_columns": dropped_columns,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_label_distribution": {
            str(k): int(v)
            for k, v in train_df[label_column].value_counts().sort_index().items()
        },
        "test_label_distribution": {
            str(k): int(v)
            for k, v in test_df[label_column].value_counts().sort_index().items()
        },
    }

    save_json(metadata, metadata_path)

    print(f"[prepare-split] saved train: {train_path}")
    print(f"[prepare-split] saved test: {test_path}")
    print(f"[prepare-split] saved metadata: {metadata_path}")


def create_random_split(
        df: pd.DataFrame,
        split_cfg: dict,
        label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = split_cfg["split_method"]

    if method["type"] != "random":
        raise NotImplementedError(f"Only random split is implemented for now. Got: {method['type']}")

    test_size = method["test_size"]
    seed = method["seed"]
    stratify_enabled = method.get("stratify", False)

    stratify = df[label_column] if stratify_enabled else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def select_final_columns(
        df: pd.DataFrame,
        dataset_cfg: dict,
        split_cfg: dict,
        feature_cfg: dict,
        label_column: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()

    dataset_drop_columns = [
        clean_column_name(c)
        for c in dataset_cfg.get("drop_columns", [])
    ]

    split_drop_columns = [
        clean_column_name(c)
        for c in split_cfg.get("drop_columns", [])
    ]

    if split_cfg.get("use_dataset_drop_columns", True):
        dropped_columns = list(dict.fromkeys(dataset_drop_columns + split_drop_columns))
    else:
        dropped_columns = list(dict.fromkeys(split_drop_columns))

    feature_definition = feature_cfg["features"]

    if isinstance(feature_definition, list):
        selected_features = [
            clean_column_name(c)
            for c in feature_definition
        ]

        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing selected features: {missing}")

    elif feature_definition == "all_numeric_except_dropped":
        blocked = set(dropped_columns)
        blocked.add(label_column)

        selected_features = [
            c
            for c in df.columns
            if c not in blocked and pd.api.types.is_numeric_dtype(df[c])
        ]

    else:
        raise ValueError(f"Unsupported feature definition: {feature_definition}")

    final_columns = selected_features + [label_column]
    final_df = df[final_columns].copy()

    final_df = final_df.replace([float("inf"), float("-inf")], pd.NA)
    final_df = final_df.dropna()

    for column in selected_features:
        final_df[column] = pd.to_numeric(final_df[column], errors="coerce")

    final_df = final_df.dropna()
    final_df[label_column] = final_df[label_column].astype(int)

    return final_df.reset_index(drop=True), selected_features, dropped_columns
