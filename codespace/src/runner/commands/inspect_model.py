import importlib
import json

from src.common.registry import load_registries


def add_inspect_model_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "inspect-model",
        help="Inspect one registered model and verify its module structure.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model ID from registries/models.json.",
    )


def run_inspect_model(args) -> None:
    _, model_registry = load_registries()

    model_id = args.model

    if model_id not in model_registry:
        raise ValueError(f"Unknown model: {model_id}")

    model_cfg = model_registry[model_id]

    print("\nModel inspection")
    print("----------------")
    print(f"model_id: {model_id}")
    print(f"name:     {model_cfg.get('name', '')}")
    print(f"type:     {model_cfg.get('type', '')}")
    print(f"module:   {model_cfg.get('module', '')}")
    print(f"ready:    {model_cfg.get('ready', False)}")
    print(f"enabled:  {model_cfg.get('enabled', True)}")

    description = model_cfg.get("description", "")
    if description:
        print(f"\ndescription:\n  {description}")

    tags = model_cfg.get("tags", [])
    if tags:
        print(f"\ntags: {tags}")

    print("\nRaw registry entry")
    print("------------------")
    print(json.dumps(model_cfg, indent=2))

    print("\nModule checks")
    print("-------------")

    module_base = model_cfg.get("module")

    if not module_base:
        print("missing module field")
        return

    check_model_module(module_base)


def check_model_module(module_base: str) -> None:
    try:
        print(f"package import:  OK  {module_base}")
    except Exception as exc:
        print(f"package import:  FAIL  {module_base}")
        print(f"  {type(exc).__name__}: {exc}")
        return

    train_module_name = f"{module_base}.train"
    evaluate_module_name = f"{module_base}.evaluate"

    try:
        train_module = importlib.import_module(train_module_name)
        print(f"train import:    OK  {train_module_name}")

        if hasattr(train_module, "train"):
            print("train function:  OK  train()")
        else:
            print("train function:  FAIL  missing train()")

    except Exception as exc:
        print(f"train import:    FAIL  {train_module_name}")
        print(f"  {type(exc).__name__}: {exc}")

    try:
        evaluate_module = importlib.import_module(evaluate_module_name)
        print(f"eval import:     OK  {evaluate_module_name}")

        if hasattr(evaluate_module, "evaluate"):
            print("eval function:   OK  evaluate()")
        else:
            print("eval function:   FAIL  missing evaluate()")

    except Exception as exc:
        print(f"eval import:     FAIL  {evaluate_module_name}")
        print(f"  {type(exc).__name__}: {exc}")
