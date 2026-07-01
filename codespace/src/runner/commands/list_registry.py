from src.common.registry import (
    load_dataset_registry,
    load_model_registry,
    load_split_registry,
    load_feature_registry,
)


def list_registry_items(title: str, registry: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for item_id, cfg in registry.items():
        name = cfg.get("name", "")
        ready = cfg.get("ready", None)
        enabled = cfg.get("enabled", True)
        description = cfg.get("description", "")

        print(f"{item_id}: {name}" if name else item_id)

        if ready is not None:
            print(f"  ready={ready}, enabled={enabled}")
        else:
            print(f"  enabled={enabled}")

        if "dataset_id" in cfg:
            print(f"  dataset_id={cfg.get('dataset_id')}")

        if "feature_set_id" in cfg:
            print(f"  feature_set_id={cfg.get('feature_set_id')}")

        if "prepared_dir" in cfg:
            print(f"  prepared_dir={cfg.get('prepared_dir')}")

        if description:
            print(f"  {description}")

    print()


def add_list_registry_parsers(subparsers) -> None:
    subparsers.add_parser("list-models", help="List registered models.")
    subparsers.add_parser("list-datasets", help="List registered datasets.")
    subparsers.add_parser("list-splits", help="List registered splits.")
    subparsers.add_parser("list-features", help="List registered feature sets.")


def run_list_models() -> None:
    registry = load_model_registry()
    list_registry_items("Available models", registry)


def run_list_datasets() -> None:
    registry = load_dataset_registry()
    list_registry_items("Available datasets", registry)


def run_list_splits() -> None:
    registry = load_split_registry()
    list_registry_items("Available splits", registry)


def run_list_features() -> None:
    registry = load_feature_registry()
    list_registry_items("Available feature sets", registry)
