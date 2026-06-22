from __future__ import annotations

import argparse

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import (
    train_eval_sklearn_model,
    write_audit_only_result,
    write_failure_result,
)


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research LightGBM runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument(
        "--source-path",
        default="external/model02_DDoS_Traffic_Research",
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
            "paper": "02",
            "repository": "DDoS_Traffic_Research",
            "variant": "lightgbm",
            "source_path": parsed.source_path,
            "n_estimators": parsed.n_estimators,
            "num_leaves": parsed.num_leaves,
            "max_depth": parsed.max_depth,
            "learning_rate": parsed.learning_rate,
            "subsample": parsed.subsample,
            "colsample_bytree": parsed.colsample_bytree,
            "reg_lambda": parsed.reg_lambda,
            "reg_alpha": parsed.reg_alpha,
            "class_weight": parsed.class_weight,
            "n_jobs": parsed.n_jobs,
        },
    )


def normalize_optional_none(value):
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def build_model(args: RunnerArgs):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    params = args.extra

    if args.problem_type == "binary":
        objective = "binary"
    else:
        objective = "multiclass"

    return LGBMClassifier(
        objective=objective,
        n_estimators=int(params["n_estimators"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        class_weight=normalize_optional_none(params["class_weight"]),
        n_jobs=int(params["n_jobs"]),
        random_state=int(args.seed),
        verbosity=-1,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model = build_model(args)

        if model is None:
            write_audit_only_result(
                args=args,
                status="needs_dependency",
                message=(
                    "LightGBM is not installed. Install it with `pip install lightgbm` "
                    "to execute this Paper 2 LightGBM runner."
                ),
                details={
                    "missing_dependency": "lightgbm",
                    "paper": "02",
                    "repository": "DDoS_Traffic_Research",
                    "variant": "lightgbm",
                    "source_path": args.extra.get("source_path"),
                },
            )
            print(f"[needs_dependency] {args.model_id}: lightgbm is not installed")
            return 0

        metrics = train_eval_sklearn_model(
            args=args,
            model=model,
            scale_numeric=False,
            model_params={
                **args.extra,
                "random_state": args.seed,
            },
        )

        metrics["paper"] = "02"
        metrics["repository"] = "DDoS_Traffic_Research"
        metrics["variant"] = "lightgbm"
        metrics["runner"] = "model02_ddos_traffic_research_lightgbm"

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