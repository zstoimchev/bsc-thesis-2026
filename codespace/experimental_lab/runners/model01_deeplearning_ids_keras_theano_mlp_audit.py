from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.paths import resolve_path
from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import write_audit_only_result, write_failure_result


def inspect_source_path(source_path: str | None) -> dict[str, Any]:
    """
    Inspect the original Paper 1 repository path for audit traceability.

    This runner does not train the Theano model. It records why the model is not
    executed and what source files are present.
    """
    if not source_path:
        return {
            "source_path_provided": False,
        }

    path = resolve_path(source_path)

    details: dict[str, Any] = {
        "source_path_provided": True,
        "source_path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "is_dir": path.is_dir() if path.exists() else False,
    }

    if path.exists() and path.is_dir():
        children = sorted(path.iterdir(), key=lambda p: p.name.lower())

        details["top_level_items_count"] = len(children)
        details["top_level_items_preview"] = [
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
            }
            for child in children[:50]
        ]

        interesting_files: list[Path] = []

        for pattern in [
            "README*",
            "requirements*.txt",
            "environment*.yml",
            "*.ipynb",
            "*.py",
            "*.md",
            "**/*theano*",
            "**/*Theano*",
            "**/*keras*",
            "**/*Keras*",
        ]:
            interesting_files.extend(path.glob(pattern))

        details["interesting_files_preview"] = [
            str(p.relative_to(path)) for p in sorted(set(interesting_files))[:80]
        ]

    return details


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 1 / DeepLearning-IDS Keras Theano MLP audit runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--source-path",
        default="external/model01_DeepLearning-IDS",
        help="Original Paper 1 repository path.",
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
            "paper": "01",
            "repository": "DeepLearning-IDS",
            "variant": "keras_theano_mlp",
            "source_path": parsed.source_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        details = inspect_source_path(args.extra.get("source_path"))

        details.update(
            {
                "paper": "01",
                "repository": "DeepLearning-IDS",
                "variant": "keras_theano_mlp",
                "reason": (
                    "The original paper reports a Keras model with Theano backend, "
                    "but Theano is obsolete and not suitable for the current unified "
                    "experiment environment. The TensorFlow Keras runner is used as "
                    "the executable Keras/MLP reproduction path."
                ),
                "recommended_executable_alternative": (
                    "model01_deeplearning_ids_keras_tf_mlp"
                ),
            }
        )

        write_audit_only_result(
            args=args,
            status="blocked",
            message=(
                "Paper 1 Keras-Theano MLP is recorded for audit only. "
                "Use the Keras-TensorFlow MLP runner for executable reproduction."
            ),
            details=details,
        )

        print(
            "[blocked] "
            f"{args.model_id}: Paper 1 Keras-Theano variant recorded as audit-only."
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())