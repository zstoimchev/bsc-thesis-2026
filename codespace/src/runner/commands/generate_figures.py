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
    "ddos_family_holdout": "DDoS family",
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

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    }
)


def add_generate_figures_parser(subparsers):
    subparsers.add_parser(
        "generate-figures",
        help="Generate the final thesis figures.",
    )


def save_figure(figure, name, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.patch.set_edgecolor("none")
    figure.patch.set_linewidth(0)
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def clean_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.margins(x=0.02)


def model_order(data, include_baseline=True):
    models = [model for model in MODEL_NAMES if model in data["model"].values]

    if not include_baseline:
        models = [model for model in models if model != "mdl00_baseline"]

    return models


def plot_metric_comparison(
        data,
        metric,
        title,
        label,
        output_name,
        include_baseline=True,
):
    selected = data[data["experiment_id"].isin(MAIN_EXPERIMENTS)]
    table = selected.pivot(index="model", columns="experiment_id", values=metric)
    table = table.reindex(
        index=model_order(selected, include_baseline=include_baseline),
        columns=MAIN_EXPERIMENTS,
    )

    table.index = [MODEL_NAMES.get(model, model) for model in table.index]
    table.columns = [EXPERIMENT_NAMES.get(experiment, experiment) for experiment in table.columns]

    axis = table.plot(kind="bar", figsize=(9, 5), width=0.88, linewidth=0)
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(label)
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=0)

    clean_axis(axis)

    axis.legend(
        title="Evaluation condition",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    save_figure(axis.get_figure(), output_name)


def plot_feature_comparison(data):
    experiments = ["loic_http_common", "loic_http_full"]
    selected = data[
        data["experiment_id"].isin(experiments)
        & (data["model"] != "mdl00_baseline")
        ]

    experiment_counts = (selected.groupby("model")["experiment_id"].nunique())

    available_models = [
        model
        for model in MODEL_NAMES
        if experiment_counts.get(model, 0) == 2
    ]

    table = selected.pivot(
        index="model",
        columns="experiment_id",
        values="f1",
    )

    table = table.reindex(
        index=available_models,
        columns=experiments,
    )

    table.index = [
        MODEL_NAMES.get(model, model)
        for model in table.index
    ]

    table.columns = [
        "11 common features",
        "57 full features",
    ]

    axis = table.plot(
        kind="bar",
        figsize=(8, 4.8),
        width=0.78,
        linewidth=0,
    )

    axis.set_title(
        "LOIC-HTTP Holdout: Feature-Set Comparison"
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("F1-score")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=0)

    clean_axis(axis)

    axis.legend(
        title="Feature set",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )

    save_figure(
        axis.get_figure(),
        "loic_feature_set_comparison",
    )


def format_duration(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m {remaining_seconds}s"

    return f"{remaining_seconds}s"


def plot_training_time(data):
    selected = data[
        (data["experiment_id"] == "hoic_common")
        & (data["model"] != "mdl00_baseline")
        & data["training_seconds"].notna()
        ].copy()

    models = model_order(
        selected,
        include_baseline=False,
    )

    selected = (
        selected
        .set_index("model")
        .reindex(models)
    )

    selected.index = [
        MODEL_NAMES.get(model, model)
        for model in selected.index
    ]

    axis = selected["training_seconds"].plot(
        kind="bar",
        figsize=(7.5, 4.8),
        width=0.65,
        linewidth=0,
    )

    axis.set_title("Training Time on the HOIC Holdout")
    axis.set_xlabel("Model")
    axis.set_ylabel("Training time in seconds")
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=0)

    clean_axis(axis)

    labels = [
        format_duration(value)
        for value in selected["training_seconds"]
    ]

    axis.bar_label(
        axis.containers[0],
        labels=labels,
        padding=3,
        fontsize=8,
    )

    save_figure(
        axis.get_figure(),
        "hoic_training_time",
    )


def plot_external_detections(data):
    selected = data[
        data["experiment_id"] == "external"
        ].copy()

    models = model_order(
        selected,
        include_baseline=True,
    )

    selected = (
        selected
        .set_index("model")
        .reindex(models)
    )

    selected.index = [
        MODEL_NAMES.get(model, model)
        for model in selected.index
    ]

    axis = selected["tp"].plot(
        kind="bar",
        figsize=(7.5, 4.8),
        width=0.65,
        linewidth=0,
    )

    axis.set_title("Malicious External Flows Correctly Detected")
    axis.set_xlabel("Model")
    axis.set_ylabel("True positives")
    axis.tick_params(axis="x", rotation=0)
    axis.ticklabel_format(style="plain", axis="y")

    clean_axis(axis)

    axis.bar_label(axis.containers[0], fmt="%.0f", padding=3, fontsize=8)

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

    figure, axis = plt.subplots(figsize=(5, 4.3))
    axis.imshow(matrix, cmap="Blues")

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Benign", "Malicious"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Benign", "Malicious"])

    labels = [
        ["True negatives", "False positives"],
        ["False negatives", "True positives"],
    ]

    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index][column_index]

            axis.text(
                column_index,
                row_index,
                (
                    f"{labels[row_index][column_index]}\n"
                    f"{value:,}"
                ),
                ha="center",
                va="center",
                fontsize=9,
            )

    save_figure(
        figure,
        output_name,
        OUTPUT_DIR / "confusion_matrices",
    )


def run_generate_figures(_args):
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Final results were not found. Run:\n"
            "python run.py summarize-results"
        )

    data = pd.read_csv(RESULTS_FILE)

    plot_metric_comparison(
        data=data,
        metric="f1",
        title="F1-score Across Evaluation Conditions",
        label="F1-score",
        output_name="f1_by_experiment",
        include_baseline=False,
    )

    plot_metric_comparison(
        data=data,
        metric="balanced_accuracy",
        title="Balanced Accuracy Across Evaluation Conditions",
        label="Balanced accuracy",
        output_name="balanced_accuracy_by_experiment",
        include_baseline=True,
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

    for (
            experiment_id,
            model,
            title,
            output_name,
    ) in confusion_matrices:
        plot_confusion_matrix(
            data=data,
            experiment_id=experiment_id,
            model=model,
            title=title,
            output_name=output_name,
        )

    print(
        f"[figures] Saved figures to: "
        f"{OUTPUT_DIR}"
    )
