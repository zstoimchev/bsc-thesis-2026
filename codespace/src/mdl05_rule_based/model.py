import numpy as np
import pandas as pd


class RuleBasedDetector:
    REQUIRED_FEATURES = [
        "flow_duration",
        "total_fwd_packets",
        "total_bwd_packets",
        "flow_packets_s",
        "flow_bytes_s",
        "syn_flag_count",
    ]

    def __init__(self) -> None:
        self.thresholds: dict[str, float] = {}
        self.feature_columns: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        missing = [feature for feature in self.REQUIRED_FEATURES if feature not in x.columns]

        if missing:
            raise ValueError(f"Rule-based detector is missing required features: {missing}")

        benign = x[y.astype(int) == 0]

        if benign.empty:
            raise ValueError("Rule-based detector requires benign training rows.")

        self.feature_columns = list(x.columns)

        self.thresholds = {
            "packet_rate": float(benign["flow_packets_s"].quantile(0.995)),
            "byte_rate": float(benign["flow_bytes_s"].quantile(0.995)),
            "syn_count": float(benign["syn_flag_count"].quantile(0.99)),
            "backward_packets": float(benign["total_bwd_packets"].quantile(0.10)),
            "short_duration": float(benign["flow_duration"].quantile(0.10)),
            "forward_packets": float(benign["total_fwd_packets"].quantile(0.99)),
        }

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not self.thresholds:
            raise ValueError("Rule-based detector has not been fitted.")

        x = x[self.feature_columns]

        high_packet_rate = (x["flow_packets_s"] > self.thresholds["packet_rate"])
        high_byte_rate = (x["flow_bytes_s"] > self.thresholds["byte_rate"])

        syn_without_response = (
                (x["syn_flag_count"] > self.thresholds["syn_count"])
                &
                (x["total_bwd_packets"] <= self.thresholds["backward_packets"])
        )

        short_high_volume = (
                (x["flow_duration"] <= self.thresholds["short_duration"])
                &
                (x["total_fwd_packets"] > self.thresholds["forward_packets"])
        )

        votes = (
                high_packet_rate.astype(int)
                + high_byte_rate.astype(int)
                + syn_without_response.astype(int)
                + short_high_volume.astype(int)
        )

        return (votes >= 1).astype(int).to_numpy()
