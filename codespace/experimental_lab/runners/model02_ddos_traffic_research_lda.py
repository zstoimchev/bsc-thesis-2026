from __future__ import annotations

import argparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import train_eval_sklearn_model, write_failure_result


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research LDA runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--solver",
        default="svd",
        choices=["svd", "lsqr", "eigen"],
        help="LDA solver.",
    )

    parser.add_argument(
        "--shrinkage",
        default=None,
        help=(
            "Shrinkage parameter. Use 'auto' or a float. "
            "Only valid for lsqr/eigen solvers."
        ),
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
            "variant": "lda",
            "source_path": parsed.source_path,
            "solver": parsed.solver,
            "shrinkage": parsed.shrinkage,
        },
    )


def parse_shrinkage(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        if value.lower() in {"none", "null"}:
            return None

        if value.lower() == "auto":
            return "auto"

        return float(value)

    return value


def build_model(args: RunnerArgs) -> LinearDiscriminantAnalysis:
    solver = str(args.extra["solver"])
    shrinkage = parse_shrinkage(args.extra["shrinkage"])

    if solver == "svd":
        # sklearn does not allow shrinkage with svd.
        return LinearDiscriminantAnalysis(
            solver=solver,
        )

    return LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model = build_model(args)

        metrics = train_eval_sklearn_model(
            args=args,
            model=model,
            scale_numeric=True,
            model_params={
                **args.extra,
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