import pandas as pd

BENIGN_LABELS = {
    "benign",
    "normal",
    "0",
    0,
}


def normalize_binary_labels(y: pd.Series) -> pd.Series:
    """
    Converts dataset labels into binary labels:
    benign/normal -> 0
    attack/malicious/anything else -> 1
    """

    def map_label(value):
        if pd.isna(value):
            return None

        normalized = str(value).strip().lower()

        if normalized in BENIGN_LABELS:
            return 0

        return 1

    mapped = y.map(map_label)

    if mapped.isna().any():
        raise ValueError("Label mapping produced missing values.")

    return mapped.astype(int)
