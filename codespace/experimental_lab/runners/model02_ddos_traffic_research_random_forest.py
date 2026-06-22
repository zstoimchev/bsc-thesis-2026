from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import train_eval_sklearn_model, write_failure_result


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research Random Forest runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", default="sqrt")
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
            "variant": "random_forest",
            "source_path": parsed.source_path,
            "n_estimators": parsed.n_estimators,
            "max_depth": parsed.max_depth,
            "min_samples_split": parsed.min_samples_split,
            "min_samples_leaf": parsed.min_samples_leaf,
            "max_features": parsed.max_features,
            "class_weight": parsed.class_weight,
            "n_jobs": parsed.n_jobs,
        },
    )


def normalize_optional_none(value):
    """
    Convert CLI string values like 'none' or 'None' to actual None.
    """
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def build_model(args: RunnerArgs) -> RandomForestClassifier:
    params = args.extra

    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=normalize_optional_none(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=normalize_optional_none(params["max_features"]),
        class_weight=normalize_optional_none(params["class_weight"]),
        n_jobs=int(params["n_jobs"]),
        random_state=int(args.seed),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model = build_model(args)

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
        metrics["variant"] = "random_forest"
        metrics["runner"] = "model02_ddos_traffic_research_random_forest"

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