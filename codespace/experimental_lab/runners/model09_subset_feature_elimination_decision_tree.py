from __future__ import annotations

import argparse
import time
from typing import Any

import joblib
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

from common.data_loading import find_label_column, split_features_label
from common.io_utils import utc_now_iso, write_json
from common.metrics import compute_classification_metrics, save_metrics_bundle
from common.paths import resolve_path
from common.preprocessing import prepare_train_test_data, save_preprocessing_artifacts
from common.runner_args import (
    RunnerArgs,
    add_common_runner_args,
    parse_drop_columns,
    runner_args_to_dict,
)
from common.runner_core import (
    load_train_test_from_args,
    write_failure_result,
)


def normalize_optional_none(value):
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None

    return value


def parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Paper 9 / ANOVA + RFE subset feature elimination + Decision Tree runner."
    )
    add_common_runner_args(parser)

    parser.add_argument(
        "--anova-k",
        type=int,
        default=20,
        help="Number of features kept by ANOVA F-test. Automatically capped to available features.",
    )
    parser.add_argument(
        "--rfe-features",
        type=int,
        default=10,
        help="Number of features kept by RFE. Automatically capped after ANOVA selection.",
    )
    parser.add_argument("--criterion", default="gini", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight", default="balanced")

    parser.add_argument(
        "--source-path",
        default="external/model09_Network-Intrusion-Detection",
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
            "paper": "09",
            "repository": "Network-Intrusion-Detection",
            "variant": "anova_rfe_decision_tree",
            "source_path": parsed.source_path,
            "anova_k": parsed.anova_k,
            "rfe_features": parsed.rfe_features,
            "criterion": parsed.criterion,
            "max_depth": parsed.max_depth,
            "min_samples_split": parsed.min_samples_split,
            "min_samples_leaf": parsed.min_samples_leaf,
            "class_weight": parsed.class_weight,
        },
    )


def run_feature_elimination_decision_tree(args: RunnerArgs) -> dict[str, Any]:
    start = time.time()

    run_dir = resolve_path(args.run_dir)
    artifact_dir = resolve_path(args.artifact_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "runner_args.json", runner_args_to_dict(args))

    loaded, train_df, test_df = load_train_test_from_args(args)

    label_col = find_label_column(train_df, args.label_col or loaded.label_col)

    X_train, y_train = split_features_label(
        train_df,
        label_col=label_col,
        drop_columns=args.drop_columns,
    )

    X_test, y_test = split_features_label(
        test_df,
        label_col=label_col,
        drop_columns=args.drop_columns,
    )

    prepared = prepare_train_test_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        problem_type=args.problem_type,
        scale_numeric=True,
    )

    X_train_array = np.asarray(prepared.X_train)
    X_test_array = np.asarray(prepared.X_test)

    original_feature_count = int(X_train_array.shape[1])

    safe_anova_k = max(1, min(int(args.extra["anova_k"]), original_feature_count))

    anova_selector = SelectKBest(
        score_func=f_classif,
        k=safe_anova_k,
    )

    X_train_anova = anova_selector.fit_transform(X_train_array, prepared.y_train)
    X_test_anova = anova_selector.transform(X_test_array)

    anova_feature_count = int(X_train_anova.shape[1])
    safe_rfe_features = max(1, min(int(args.extra["rfe_features"]), anova_feature_count))

    rfe_selector = None

    if safe_rfe_features < anova_feature_count:
        rfe_estimator = DecisionTreeClassifier(
            criterion=str(args.extra["criterion"]),
            max_depth=args.extra["max_depth"],
            min_samples_split=int(args.extra["min_samples_split"]),
            min_samples_leaf=int(args.extra["min_samples_leaf"]),
            class_weight=normalize_optional_none(args.extra["class_weight"]),
            random_state=int(args.seed),
        )

        rfe_selector = RFE(
            estimator=rfe_estimator,
            n_features_to_select=safe_rfe_features,
            step=1,
        )

        X_train_selected = rfe_selector.fit_transform(X_train_anova, prepared.y_train)
        X_test_selected = rfe_selector.transform(X_test_anova)
    else:
        X_train_selected = X_train_anova
        X_test_selected = X_test_anova

    classifier = DecisionTreeClassifier(
        criterion=str(args.extra["criterion"]),
        max_depth=args.extra["max_depth"],
        min_samples_split=int(args.extra["min_samples_split"]),
        min_samples_leaf=int(args.extra["min_samples_leaf"]),
        class_weight=normalize_optional_none(args.extra["class_weight"]),
        random_state=int(args.seed),
    )

    classifier.fit(X_train_selected, prepared.y_train)

    y_pred = classifier.predict(X_test_selected)

    y_proba = None
    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(X_test_selected)

    metrics = compute_classification_metrics(
        y_true=prepared.y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        label_mapping=prepared.label_mapping,
        problem_type=args.problem_type,
    )

    selector_bundle_path = artifact_dir / "feature_selectors.joblib"
    model_path = artifact_dir / "decision_tree.joblib"

    joblib.dump(
        {
            "anova_selector": anova_selector,
            "rfe_selector": rfe_selector,
            "original_feature_count": original_feature_count,
            "anova_feature_count": anova_feature_count,
            "final_feature_count": int(X_train_selected.shape[1]),
        },
        selector_bundle_path,
    )

    joblib.dump(classifier, model_path)

    save_preprocessing_artifacts(str(artifact_dir), prepared)

    duration = time.time() - start

    metrics.update(
        {
            "run_started_utc": utc_now_iso(),
            "duration_seconds": round(float(duration), 4),
            "status": "success",
            "stage": args.stage,
            "mode": args.mode,
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "split_id": args.split_id,
            "runner": "model09_subset_feature_elimination_decision_tree",
            "paper": "09",
            "repository": "Network-Intrusion-Detection",
            "variant": "anova_rfe_decision_tree",
            "problem_type": args.problem_type,
            "label_col": label_col,
            "drop_columns": args.drop_columns,
            "original_feature_count": original_feature_count,
            "requested_anova_k": int(args.extra["anova_k"]),
            "actual_anova_k": safe_anova_k,
            "anova_feature_count": anova_feature_count,
            "requested_rfe_features": int(args.extra["rfe_features"]),
            "actual_rfe_features": safe_rfe_features,
            "final_feature_count": int(X_train_selected.shape[1]),
            "rfe_applied": bool(rfe_selector is not None),
            "criterion": str(args.extra["criterion"]),
            "max_depth": args.extra["max_depth"],
            "min_samples_split": int(args.extra["min_samples_split"]),
            "min_samples_leaf": int(args.extra["min_samples_leaf"]),
            "class_weight": args.extra["class_weight"],
            "selector_bundle_path": str(selector_bundle_path),
            "model_artifact_path": str(model_path),
            "artifact_dir": str(artifact_dir),
            "run_dir": str(run_dir),
        }
    )

    saved_metrics = save_metrics_bundle(run_dir, metrics)

    metrics_out = resolve_path(args.metrics_out)
    if metrics_out != run_dir / "metrics.json":
        write_json(metrics_out, saved_metrics)

    return saved_metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        metrics = run_feature_elimination_decision_tree(args)

        print(
            f"[success] {args.model_id} on {args.dataset_id}: "
            f"accuracy={metrics.get('accuracy')}, "
            f"f1_macro={metrics.get('f1_macro')}, "
            f"final_features={metrics.get('final_feature_count')}"
        )

        return 0

    except Exception as exc:
        write_failure_result(args, exc)
        print(f"[failed] {args.model_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())