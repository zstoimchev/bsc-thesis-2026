import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    split_strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if split_strategy != "random":
        raise NotImplementedError(
            f"Split strategy '{split_strategy}' is not implemented yet."
        )

    stratify = y if y.nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }