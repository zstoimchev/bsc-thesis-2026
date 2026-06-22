from __future__ import annotations

import argparse

from sklearn.neighbors import KNeighborsClassifier

from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import train_eval_sklearn_model, write_failure_result


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 2 / DDoS_Traffic_Research KNN runner."
    )
    add_common_runner_args(parser)

    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--weights", default="distance", choices=["uniform", "distance"])
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--leaf-size", type=int, default=30)
    parser.add_argument("--p", type=int, default=2)
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
            "variant": "knn",
            "source_path": parsed.source_path,
            "n_neighbors": parsed.n_neighbors,
            "weights": parsed.weights,
            "algorithm": parsed.algorithm,
            "leaf_size": parsed.leaf_size,
            "p": parsed.p,
            "n_jobs": parsed.n_jobs,
        },
    )


def build_model(args: RunnerArgs) -> KNeighborsClassifier:
    params = args.extra

    return KNeighborsClassifier(
        n_neighbors=int(params["n_neighbors"]),
        weights=str(params["weights"]),
        algorithm=str(params["algorithm"]),
        leaf_size=int(params["leaf_size"]),
        p=int(params["p"]),
        n_jobs=int(params["n_jobs"]),
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

        metrics["paper"] = "02"
        metrics["repository"] = "DDoS_Traffic_Research"
        metrics["variant"] = "knn"
        metrics["runner"] = "model02_ddos_traffic_research_knn"

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