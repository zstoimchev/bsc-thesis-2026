from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass
class RunnerArgs:
    """
    Standard argument object used by all runners.

    Every runner should accept the same arguments so orchestrate.py can call all of
    them in the same way.
    """

    mode: str
    stage: str
    model_id: str
    dataset_id: str
    split_id: str | None
    dataset_path: str | None
    train_path: str | None
    test_path: str | None
    label_col: str | None
    drop_columns: list[str]
    problem_type: str
    artifact_dir: str
    run_dir: str
    metrics_out: str
    nrows: int | None
    seed: int
    extra: dict[str, Any]


def parse_drop_columns(value: str | None) -> list[str]:
    """
    Parse comma-separated drop column argument.

    Example:
        "src_ip,dst_ip,flow_id" -> ["src_ip", "dst_ip", "flow_id"]
    """
    if value is None:
        return []

    value = value.strip()

    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def add_common_runner_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard runner arguments to an ArgumentParser.
    """
    parser.add_argument(
        "--mode",
        default="train-eval",
        choices=["train", "eval", "train-eval", "audit"],
        help="Runner execution mode.",
    )

    parser.add_argument(
        "--stage",
        default="unified",
        choices=["native", "unified", "audit"],
        help="Experiment stage.",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model registry id.",
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset registry id.",
    )

    parser.add_argument(
        "--split-id",
        default=None,
        help="Split registry id. Optional for predefined train/test datasets.",
    )

    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Single-file dataset path, used for generated split datasets.",
    )

    parser.add_argument(
        "--train-path",
        default=None,
        help="Predefined train dataset path.",
    )

    parser.add_argument(
        "--test-path",
        default=None,
        help="Predefined test dataset path.",
    )

    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column name.",
    )

    parser.add_argument(
        "--drop-columns",
        default="",
        help="Comma-separated columns to remove before training.",
    )

    parser.add_argument(
        "--problem-type",
        default="binary",
        choices=["binary", "multiclass"],
        help="Classification problem type.",
    )

    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory where trained model/preprocessing artifacts should be saved.",
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory for this run.",
    )

    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Path where metrics.json must be written.",
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for fast testing. Use only for debugging.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser


def parse_runner_args(argv: list[str] | None = None) -> RunnerArgs:
    """
    Parse standard runner arguments.

    Unknown args are collected into RunnerArgs.extra so specific runners can add
    model-specific options later without breaking orchestrate.py.
    """
    parser = argparse.ArgumentParser()
    add_common_runner_args(parser)

    args, unknown = parser.parse_known_args(argv)

    extra: dict[str, Any] = {}

    # Store unknown args as a simple list for now.
    # Later, if needed, individual runners can parse them more strictly.
    if unknown:
        extra["unknown_args"] = unknown

    return RunnerArgs(
        mode=args.mode,
        stage=args.stage,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        dataset_path=args.dataset_path,
        train_path=args.train_path,
        test_path=args.test_path,
        label_col=args.label_col,
        drop_columns=parse_drop_columns(args.drop_columns),
        problem_type=args.problem_type,
        artifact_dir=args.artifact_dir,
        run_dir=args.run_dir,
        metrics_out=args.metrics_out,
        nrows=args.nrows,
        seed=args.seed,
        extra=extra,
    )


def runner_args_to_dict(args: RunnerArgs) -> dict[str, Any]:
    """
    Convert RunnerArgs dataclass into JSON-safe dictionary.
    """
    return {
        "mode": args.mode,
        "stage": args.stage,
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "split_id": args.split_id,
        "dataset_path": args.dataset_path,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "label_col": args.label_col,
        "drop_columns": args.drop_columns,
        "problem_type": args.problem_type,
        "artifact_dir": args.artifact_dir,
        "run_dir": args.run_dir,
        "metrics_out": args.metrics_out,
        "nrows": args.nrows,
        "seed": args.seed,
        "extra": args.extra,
    }