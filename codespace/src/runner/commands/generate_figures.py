import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.runner.constants import PROJECT_ROOT

RESULTS_FILE = (
        PROJECT_ROOT
        / "results"
        / "final"
        / "final_results.csv"
)

OUTPUT_DIR = (
        PROJECT_ROOT
        / "results"
        / "final"
        / "figures"
)

MODEL_NAMES = {
    "mdl00_baseline": "Baseline",
    "mdl01_xgboost": "XGBoost",
    "mdl02_mlp": "MLP",
    "mdl03_gru": "GRU",
    "mdl04_transformer": "Transformer",
    "mdl05_rule_based": "Rule-based",
}

EXPERIMENT_NAMES = {
    "random_common": "Random split",
    "ddos_family_holdout": "DDoS holdout",
    "loic_http_common": "LOIC-HTTP",
    "hoic_common": "HOIC",
    "external": "External",
    "loic_http_full": "LOIC-HTTP full",
}

MAIN_EXPERIMENTS = [
    "random_common",
    "ddos_family_holdout",
    "loic_http_common",
    "hoic_common",
    "external",
]


def add_generate_figures_parser(subparsers):
    subparsers.add_parser(
        "generate-figures",
        help="Generate the final thesis figures.",
    )


def save_figure(figure, name, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def model_order(data):
    return [model for model in MODEL_NAMES if model in data["model"].values]


def plot_metric_comparison(data, metric, title, output_name):
    selected = data[data["experiment_id"].isin(MAIN_EXPERIMENTS)]

    table = selected.pivot(index="model", columns="experiment_id", values=metric)
    table = table.reindex(index=model_order(selected), columns=MAIN_EXPERIMENTS)
    table.index = [MODEL_NAMES.get(model, model) for model in table.index]
    table.columns = [EXPERIMENT_NAMES.get(experiment, experiment) for experiment in table.columns]

    axis = table.plot(kind="bar", figsize=(11, 6))
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(axis.get_figure(), output_name)


def plot_feature_comparison(data):
    experiments = ["loic_http_common", "loic_http_full"]
    selected = data[data["experiment_id"].isin(experiments)]
    table = selected.pivot(index="model", columns="experiment_id", values="f1")
    table = table.reindex(index=model_order(selected), columns=experiments)
    table.index = [MODEL_NAMES.get(model, model) for model in table.index]
    table.columns = ["11 common features", "57 full features"]
    axis = table.plot(kind="bar", figsize=(9, 5))
    axis.set_title("LOIC-HTTP Holdout: Feature-Set Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("F1-score")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)

    save_figure(axis.get_figure(), "loic_feature_set_comparison")


def plot_training_time(data):
    selected = data[
        (data["experiment_id"] == "hoic_common")
        & (data["model"] != "mdl00_baseline")
        & data["training_seconds"].notna()
        ].copy()

    selected = selected.set_index("model")
    selected = selected.reindex(model_order(selected.reset_index()))
    selected.index = [MODEL_NAMES.get(model, model) for model in selected.index]
    axis = selected["training_seconds"].plot(kind="bar", figsize=(8, 5))

    axis.set_title("Training Time on the HOIC Holdout")
    axis.set_xlabel("Model")
    axis.set_ylabel("Training time in seconds")
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)

    save_figure(axis.get_figure(), "hoic_training_time")


def plot_external_detections(data):
    selected = data[data["experiment_id"] == "external"].copy()

    selected = selected.set_index("model")
    selected = selected.reindex(model_order(selected.reset_index()))
    selected.index = [MODEL_NAMES.get(model, model) for model in selected.index]
    axis = selected["tp"].plot(kind="bar", figsize=(8, 5))
    axis.set_title("Malicious External Flows Correctly Detected")
    axis.set_xlabel("Model")
    axis.set_ylabel("True positives")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=3)

    save_figure(axis.get_figure(), "external_detected_attacks")


def plot_confusion_matrix(
        data,
        experiment_id,
        model,
        title,
        output_name,
):
    selected = data[(data["experiment_id"] == experiment_id) & (data["model"] == model)]

    if selected.empty:
        print(f"[figures] Skipping missing result: {experiment_id}/{model}")
        return

    row = selected.iloc[0]

    matrix = [
        [int(row["tn"]), int(row["fp"])],
        [int(row["fn"]), int(row["tp"])],
    ]

    figure, axis = plt.subplots(figsize=(5, 4))

    image = axis.imshow(matrix)

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Benign", "Malicious"])

    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Benign", "Malicious"])

    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index][column_index]:,}",
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    save_figure(figure, output_name, OUTPUT_DIR / "confusion_matrices")


def run_generate_figures(args):
    if not RESULTS_FILE.exists():
        raise FileNotFoundError("Final results were not found. Run: python run.py summarize-results")

    data = pd.read_csv(RESULTS_FILE)

    plot_metric_comparison(
        data,
        metric="f1",
        title="F1-score Across Evaluation Conditions",
        output_name="f1_by_experiment",
    )

    plot_metric_comparison(
        data,
        metric="balanced_accuracy",
        title="Balanced Accuracy Across Evaluation Conditions",
        output_name="balanced_accuracy_by_experiment",
    )

    plot_feature_comparison(data)
    plot_training_time(data)
    plot_external_detections(data)

    confusion_matrices = [
        (
            "loic_http_common",
            "mdl03_gru",
            "GRU on the LOIC-HTTP Holdout",
            "confusion_matrix_loic_http_gru",
        ),
        (
            "hoic_common",
            "mdl01_xgboost",
            "XGBoost on the HOIC Holdout",
            "confusion_matrix_hoic_xgboost",
        ),
        (
            "ddos_family_holdout",
            "mdl01_xgboost",
            "XGBoost on the DDoS-Family Holdout",
            "confusion_matrix_ddos_family_xgboost",
        ),
        (
            "external",
            "mdl02_mlp",
            "MLP on the External Dataset",
            "confusion_matrix_external_mlp",
        ),
    ]

    for (experiment_id, model, title, output_name) in confusion_matrices:
        plot_confusion_matrix(data, experiment_id, model, title, output_name)

    print(f"[figures] Saved figures to: {OUTPUT_DIR}")
