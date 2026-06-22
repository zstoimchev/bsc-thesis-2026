from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.paths import resolve_path
from common.runner_args import RunnerArgs, add_common_runner_args, parse_drop_columns
from common.runner_core import write_audit_only_result, write_failure_result


def inspect_source_path(source_path: str | None) -> dict[str, Any]:
    """
    Inspect an optional external repository/path.

    This does not execute anything. It only records whether the path exists and
    gives a small file preview for audit purposes.
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

        interesting_files = []
        for pattern in [
            "README*",
            "requirements*.txt",
            "environment*.yml",
            "*.ipynb",
            "*.py",
            "*.md",
        ]:
            interesting_files.extend(path.glob(pattern))

        details["interesting_files_preview"] = [
            str(p.relative_to(path)) for p in sorted(set(interesting_files))[:50]
        ]

    if path.exists() and path.is_file():
        details["file_size_bytes"] = path.stat().st_size

    return details


def parse_args(argv: list[str] | None = None) -> tuple[RunnerArgs, str, str, str | None]:
    """
    Parse audit runner arguments.

    This runner accepts all common runner args plus:
    - --audit-status
    - --message
    - --source-path
    """
    parser = argparse.ArgumentParser(description="Audit-only runner for blocked/unready repositories.")
    add_common_runner_args(parser)

    parser.add_argument(
        "--audit-status",
        default="audit_only",
        choices=["audit_only", "blocked", "needs_data", "needs_dependency", "skipped"],
        help="Structured audit status to write into metrics.json.",
    )

    parser.add_argument(
        "--message",
        default="Audit-only runner. No training/evaluation executed.",
        help="Human-readable audit message.",
    )

    parser.add_argument(
        "--source-path",
        default=None,
        help="Optional external repository/model path to inspect.",
    )

    parsed = parser.parse_args(argv)

    runner_args = RunnerArgs(
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
            "audit_status": parsed.audit_status,
            "message": parsed.message,
            "source_path": parsed.source_path,
        },
    )

    return runner_args, parsed.audit_status, parsed.message, parsed.source_path


def main(argv: list[str] | None = None) -> int:
    args, audit_status, message, source_path = parse_args(argv)

    try:
        details = inspect_source_path(source_path)

        write_audit_only_result(
            args=args,
            status=audit_status,
            message=message,
            details=details,
        )

        print(f"[{audit_status}] {args.model_id}: {message}")
        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())