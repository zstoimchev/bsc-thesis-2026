from collections import Counter

import numpy as np
import pandas as pd


class MajorityClassBaseline:
    def __init__(self) -> None:
        self.label_counts = Counter()
        self.majority_class: int | None = None
        self.feature_columns: list[str] | None = None

    def partial_fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if self.feature_columns is None:
            self.feature_columns = list(x.columns)
        elif self.feature_columns != list(x.columns):
            raise ValueError(
                "Feature columns changed between chunks. "
                "This means preprocessing is inconsistent."
            )

        self.label_counts.update(y.astype(int).tolist())

    def finalize(self) -> None:
        if not self.label_counts:
            raise ValueError("Cannot finalize baseline model without labels.")

        self.majority_class = int(self.label_counts.most_common(1)[0][0])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.majority_class is None:
            raise ValueError("Model has not been finalized.")

        return np.full(
            shape=len(x),
            fill_value=self.majority_class,
            dtype=int,
        )
