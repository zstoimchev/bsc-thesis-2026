from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.paths import resolve_path
from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import write_audit_only_result, write_failure_result


def inspect_source_path(source_path: str | None) -> dict[str, Any]:
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
            for child in children[:80]
        ]

        interesting_files: list[Path] = []

        for pattern in [
            "README*",
            "requirements*.txt",
            "environment*.yml",
            "*.ipynb",
            "*.py",
            "*.md",
            "**/*.ipynb",
            "**/*.py",
            "**/*dnn*",
            "**/*DNN*",
            "**/*mlp*",
            "**/*MLP*",
            "**/*dense*",
            "**/*Dense*",
            "**/*keras*",
            "**/*tensorflow*",
        ]:
            interesting_files.extend(path.glob(pattern))

        details["interesting_files_preview"] = [
            str(p.relative_to(path)) for p in sorted(set(interesting_files))[:120]
        ]

    if path.exists() and path.is_file():
        details["file_size_bytes"] = path.stat().st_size

    return details


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 14 / DNN IDS remaining variants audit runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--source-path",
        default="external/model14_dnn_ids",
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
            "paper": "14",
            "repository": "dnn_ids",
            "variant": "remaining_dnn_mlp_variants_audit",
            "source_path": parsed.source_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        details = inspect_source_path(args.extra.get("source_path"))

        details.update(
            {
                "paper": "14",
                "repository": "dnn_ids",
                "variant": "remaining_dnn_mlp_variants_audit",
                "trainable": False,
                "implemented_trainable_representatives": [
                    "model14_dnn_ids_mlp",
                ],
                "remaining_variants_recorded_here": [
                    "alternative_dense_layer_depths",
                    "alternative_dropout_values",
                    "alternative_optimizer_or_learning_rate_settings",
                    "paper_specific_mlp_hyperparameter_variants",
                ],
                "reason": (
                    "Paper 14 is represented by an executable DNN/MLP runner. "
                    "This audit file records remaining dense-network variants and "
                    "hyperparameter configurations without duplicating nearly identical "
                    "MLP code."
                ),
            }
        )

        write_audit_only_result(
            args=args,
            status="audit_only",
            message=(
                "Paper 14 remaining DNN/MLP variants recorded as audit-only. "
                "The executable MLP representative is implemented separately."
            ),
            details=details,
        )

        print(
            "[audit_only] "
            f"{args.model_id}: Paper 14 remaining variants recorded."
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())