from __future__ import annotations

import argparse
from typing import Any

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import (
    train_eval_sklearn_model,
    write_audit_only_result,
    write_failure_result,
)


def normalize_optional_none(value):
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 11 / Network-Intrusion-Detection-System XGBoost runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--tree-method", default="hist")

    parser.add_argument(
        "--source-path",
        default="external/model11_Network-Intrusion-Detection-System",
        help="Optional original repository path for audit traceability.",
    )

    parsed = parser.parse_args(argv)

    return RunnerArgs(
        mode=parsed.mode,
        stage=parsed.stage,
        model_id=parsed.model_id,
        dataset_id=parsed.dataset_id,
        split_id=parsed.split_id,
        dataset_path=parsed.dataset_path,
        train_path=parsed.train_path,
        test_path=parsed.test_path,
        label_col=parsed.label_col,
        drop_columns=parse_drop_columns(parsed.drop_columns),
        problem_type=parsed.problem_type,
        artifact_dir=parsed.artifact_dir,
        run_dir=parsed.run_dir,
        metrics_out=parsed.metrics_out,
        nrows=parsed.nrows,
        seed=parsed.seed,
        extra={
            "paper": "11",
            "repository": "Network-Intrusion-Detection-System",
            "variant": "xgboost",
            "source_path": parsed.source_path,
            "n_estimators": parsed.n_estimators,
            "max_depth": parsed.max_depth,
            "learning_rate": parsed.learning_rate,
            "subsample": parsed.subsample,
            "colsample_bytree": parsed.colsample_bytree,
            "reg_lambda": parsed.reg_lambda,
            "reg_alpha": parsed.reg_alpha,
            "tree_method": parsed.tree_method,
        },
    )


def build_xgboost_model(args: RunnerArgs):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    if args.problem_type == "binary":
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"

    return XGBClassifier(
        n_estimators=int(args.extra["n_estimators"]),
        max_depth=int(args.extra["max_depth"]),
        learning_rate=float(args.extra["learning_rate"]),
        subsample=float(args.extra["subsample"]),
        colsample_bytree=float(args.extra["colsample_bytree"]),
        reg_lambda=float(args.extra["reg_lambda"]),
        reg_alpha=float(args.extra["reg_alpha"]),
        tree_method=str(args.extra["tree_method"]),
        objective=objective,
        eval_metric=eval_metric,
        random_state=int(args.seed),
        n_jobs=-1,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model = build_xgboost_model(args)

        if model is None:
            metrics = write_audit_only_result(
                args=args,
                status="needs_dependency",
                message=(
                    "XGBoost is not installed. Install it with `pip install xgboost` "
                    "to execute this Paper 11 XGBoost runner."
                ),
                details={
                    "missing_dependency": "xgboost",
                    "paper": "11",
                    "repository": "Network-Intrusion-Detection-System",
                    "variant": "xgboost",
                    "source_path": args.extra.get("source_path"),
                },
            )

            print(f"[{metrics.get('status')}] {args.model_id}: {metrics.get('message')}")
            return 0

        metrics = train_eval_sklearn_model(
            args=args,
            model=model,
            scale_numeric=True,
            model_params={
                "paper": "11",
                "repository": "Network-Intrusion-Detection-System",
                "variant": "xgboost",
                "n_estimators": int(args.extra["n_estimators"]),
                "max_depth": int(args.extra["max_depth"]),
                "learning_rate": float(args.extra["learning_rate"]),
                "subsample": float(args.extra["subsample"]),
                "colsample_bytree": float(args.extra["colsample_bytree"]),
                "reg_lambda": float(args.extra["reg_lambda"]),
                "reg_alpha": float(args.extra["reg_alpha"]),
                "tree_method": str(args.extra["tree_method"]),
                "source_path": args.extra.get("source_path"),
            },
        )

        metrics.update(
            {
                "paper": "11",
                "repository": "Network-Intrusion-Detection-System",
                "variant": "xgboost",
                "source_path": args.extra.get("source_path"),
            }
        )

        print(
            f"[success] {args.model_id} on {args.dataset_id}: "
            f"accuracy={metrics.get('accuracy')}, "
            f"f1_macro={metrics.get('f1_macro')}"
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())