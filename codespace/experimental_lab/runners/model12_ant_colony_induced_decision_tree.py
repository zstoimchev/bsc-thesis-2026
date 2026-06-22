from __future__ import annotations

import argparse
from typing import Any

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import train_eval_sklearn_model, write_failure_result


def normalize_optional_none(value):
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def parse_feature_k(value: str):
    value = value.strip().lower()

    if value in {"all", "none"}:
        return "all"

    return int(value)


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 12 / Ant Colony Induced Decision Trees surrogate runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--feature-k",
        default="all",
        help="Number of features kept before the decision tree. Use 'all' to disable selection.",
    )
    parser.add_argument("--criterion", default="gini", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight", default="balanced")

    parser.add_argument(
        "--source-path",
        default="external/model12_NSLKDD-Dataset",
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
            "paper": "12",
            "repository": "NSLKDD-Dataset",
            "variant": "ant_colony_induced_decision_tree_surrogate",
            "source_path": parsed.source_path,
            "feature_k": parsed.feature_k,
            "criterion": parsed.criterion,
            "max_depth": parsed.max_depth,
            "min_samples_split": parsed.min_samples_split,
            "min_samples_leaf": parsed.min_samples_leaf,
            "class_weight": parsed.class_weight,
        },
    )


def build_decision_tree_pipeline(args: RunnerArgs) -> Pipeline:
    feature_k = parse_feature_k(str(args.extra["feature_k"]))

    selector = SelectKBest(
        score_func=f_classif,
        k=feature_k,
    )

    tree = DecisionTreeClassifier(
        criterion=str(args.extra["criterion"]),
        max_depth=args.extra["max_depth"],
        min_samples_split=int(args.extra["min_samples_split"]),
        min_samples_leaf=int(args.extra["min_samples_leaf"]),
        class_weight=normalize_optional_none(args.extra["class_weight"]),
        random_state=int(args.seed),
    )

    return Pipeline(
        steps=[
            ("feature_selector", selector),
            ("decision_tree", tree),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model = build_decision_tree_pipeline(args)

        metrics = train_eval_sklearn_model(
            args=args,
            model=model,
            scale_numeric=True,
            model_params={
                "paper": "12",
                "repository": "NSLKDD-Dataset",
                "variant": "ant_colony_induced_decision_tree_surrogate",
                "note": (
                    "This runner is a runnable decision-tree surrogate for the "
                    "Ant Colony Induced Decision Trees paper. It does not implement "
                    "the full ATMa ant-colony optimization algorithm."
                ),
                "feature_k": args.extra["feature_k"],
                "criterion": str(args.extra["criterion"]),
                "max_depth": args.extra["max_depth"],
                "min_samples_split": int(args.extra["min_samples_split"]),
                "min_samples_leaf": int(args.extra["min_samples_leaf"]),
                "class_weight": args.extra["class_weight"],
                "source_path": args.extra.get("source_path"),
            },
        )

        metrics.update(
            {
                "paper": "12",
                "repository": "NSLKDD-Dataset",
                "variant": "ant_colony_induced_decision_tree_surrogate",
                "source_path": args.extra.get("source_path"),
                "exact_atma_implementation": False,
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