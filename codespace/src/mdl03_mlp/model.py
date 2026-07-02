from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=2048,
                    learning_rate_init=0.001,
                    max_iter=20,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=seed,
                    verbose=True,
                ),
            ),
        ]
    )
