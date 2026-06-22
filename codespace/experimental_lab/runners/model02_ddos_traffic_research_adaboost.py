from __future__ import annotations

import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import train_eval_sklearn_model, write_failure_result


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research AdaBoost runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument(
        "--base-max-depth",
        type=int,
        default=1,
        help="Maximum depth of the weak decision-tree learner.",
    )

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
            "variant": "adaboost",
            "source_path": parsed.source_path,
            "n_estimators": parsed.n_estimators,
            "learning_rate": parsed.learning_rate,
            "base_max_depth": parsed.base_max_depth,
        },
    )


def build_model(args: RunnerArgs) -> AdaBoostClassifier:
    params = args.extra

    base_estimator = DecisionTreeClassifier(
        max_depth=int(params["base_max_depth"]),
        random_state=int(args.seed),
    )

    return AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
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