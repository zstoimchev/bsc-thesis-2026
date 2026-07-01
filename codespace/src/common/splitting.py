import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame,
    label_column: str,
    split_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method = split_cfg["split_method"]
    split_type = method["type"]

    if split_type == "random":
        return random_split(
            df=df,
            label_column=label_column,
            test_size=method.get("test_size", 0.2),
            seed=method.get("seed", 42),
            stratify_enabled=method.get("stratify", False),
        )

    raise NotImplementedError(
        f"Split type '{split_type}' is not implemented yet."
    )


def random_split(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_enabled and df[label_column].nunique() > 1:
        stratify = df[label_column]
    else:
        stratify = None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)