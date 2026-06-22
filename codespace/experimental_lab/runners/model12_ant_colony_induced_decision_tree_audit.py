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
            "**/*ant*",
            "**/*Ant*",
            "**/*colony*",
            "**/*Colony*",
            "**/*atma*",
            "**/*ATMa*",
            "**/*tree*",
            "**/*Tree*",
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
        description="Paper 12 / Ant Colony Induced Decision Trees audit runner."
    )
    add_common_runner_args(parser)

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
            "variant": "atma_ant_colony_decision_tree_audit",
            "source_path": parsed.source_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        details = inspect_source_path(args.extra.get("source_path"))

        details.update(
            {
                "paper": "12",
                "repository": "NSLKDD-Dataset",
                "variant": "atma_ant_colony_decision_tree_audit",
                "trainable": False,
                "implemented_trainable_representatives": [
                    "model12_ant_colony_induced_decision_tree",
                ],
                "exact_atma_implementation": False,
                "surrogate_implementation": (
                    "model12_ant_colony_induced_decision_tree implements a runnable "
                    "feature-selection + Decision Tree surrogate, not the full ATMa "
                    "ant-colony optimization procedure."
                ),
                "remaining_variants_recorded_here": [
                    "exact_atma_ant_colony_tree_induction",
                    "pheromone_based_rule_or_tree_construction",
                    "paper_specific_ant_colony_parameters",
                    "original_nsl_kdd_specific_preprocessing",
                ],
                "reason": (
                    "Paper 12 describes ant-colony-induced decision trees. Because the "
                    "exact ATMa algorithm is specialized and not central to the final "
                    "DDoS comparison, the executable runner uses a transparent Decision "
                    "Tree surrogate. This audit file preserves traceability and clearly "
                    "marks the exact ATMa implementation as not reimplemented."
                ),
            }
        )

        write_audit_only_result(
            args=args,
            status="audit_only",
            message=(
                "Paper 12 exact ATMa ant-colony decision-tree method recorded as "
                "audit-only. A runnable Decision Tree surrogate is implemented separately."
            ),
            details=details,
        )

        print(
            "[audit_only] "
            f"{args.model_id}: Paper 12 exact ATMa method recorded."
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())