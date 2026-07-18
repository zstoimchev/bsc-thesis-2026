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

    if split_type == "attack_family_holdout":
        return attack_family_holdout_split(
            df=df,
            label_column=label_column,
            group_column=method["group_column"],
            holdout_value=method["holdout_value"],
            benign_test_size=method.get("benign_test_size", 0.2),
            seed=method.get("seed", 42),
        )

    if split_type == "external_full":
        return external_full_split(df)

    raise NotImplementedError(f"Split type '{split_type}' is not implemented.")


def random_split(
        df: pd.DataFrame,
        label_column: str,
        test_size: float = 0.2,
        seed: int = 42,
        stratify_enabled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None

    if stratify_enabled and df[label_column].nunique() > 1:
        stratify = df[label_column]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def attack_family_holdout_split(
        df: pd.DataFrame,
        label_column: str,
        group_column: str,
        holdout_value: str,
        benign_test_size: float,
        seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column not in df.columns:
        raise ValueError(f"Holdout group column not found: {group_column}")

    groups = df[group_column].astype(str).str.strip().str.lower()
    holdout = str(holdout_value).strip().lower()

    holdout_mask = groups == holdout

    if not holdout_mask.any():
        raise ValueError(f"No rows found where {group_column}={holdout_value}")

    holdout_attacks = df[holdout_mask]

    if not (holdout_attacks[label_column] == 1).all():
        raise ValueError("The held-out attack family contains non-malicious rows.")

    benign = df[df[label_column] == 0]
    other_attacks = df[(df[label_column] == 1) & ~holdout_mask]

    benign_train, benign_test = train_test_split(
        benign,
        test_size=benign_test_size,
        random_state=seed,
        shuffle=True,
    )

    train_df = pd.concat(
        [benign_train, other_attacks],
        ignore_index=True,
    )

    test_df = pd.concat(
        [benign_test, holdout_attacks],
        ignore_index=True,
    )

    train_df = train_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return train_df, test_df


def external_full_split(
        df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[0:0].copy()
    test_df = df.copy()

    return train_df, test_df.reset_index(drop=True)
