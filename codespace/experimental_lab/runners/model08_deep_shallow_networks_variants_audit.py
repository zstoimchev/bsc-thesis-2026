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
            "**/*dbn*",
            "**/*DBN*",
            "**/*autoencoder*",
            "**/*Autoencoder*",
            "**/*ae*",
            "**/*deep*",
            "**/*Deep*",
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
        description="Paper 8 / Deep-Shallow remaining variants audit runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--source-path",
        default="external/model08_deep_shallow_networks",
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
            "paper": "08",
            "repository": "deep_shallow_networks",
            "variant": "remaining_dbn_autoencoder_variants_audit",
            "source_path": parsed.source_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        details = inspect_source_path(args.extra.get("source_path"))

        details.update(
            {
                "paper": "08",
                "repository": "deep_shallow_networks",
                "variant": "remaining_dbn_autoencoder_variants_audit",
                "trainable": False,
                "implemented_trainable_representatives": [
                    "model08_deep_shallow_networks_stacked_autoencoder_mlp",
                ],
                "remaining_variants_recorded_here": [
                    "dbn_style_representation_learning_if_present",
                    "shallow_network_baselines_if_present",
                    "minor_autoencoder_hyperparameter_variants",
                    "paper_specific_deep_shallow_configurations",
                ],
                "reason": (
                    "Paper 8 belongs to the DBN/autoencoder/deep-shallow family. "
                    "The executable representative is implemented as stacked "
                    "autoencoder feature learning followed by an MLP classifier. "
                    "This audit runner records remaining variants without creating "
                    "duplicated or obsolete code."
                ),
            }
        )

        write_audit_only_result(
            args=args,
            status="audit_only",
            message=(
                "Paper 8 remaining DBN/deep-shallow variants recorded as audit-only. "
                "The stacked autoencoder + MLP representative is implemented separately."
            ),
            details=details,
        )

        print(
            "[audit_only] "
            f"{args.model_id}: Paper 8 remaining variants recorded."
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())