import numpy as np
import pandas as pd


class RuleBasedDetector:
    REQUIRED_FEATURES = [
        "flow_duration",
        "total_fwd_packets",
        "total_bwd_packets",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "flow_packets_s",
        "flow_bytes_s",
        "syn_flag_count",
    ]

    def __init__(
        self,
        anomaly_quantile: float = 0.99,
        required_votes: int = 1,
    ) -> None:
        if not 0 < anomaly_quantile < 1:
            raise ValueError(
                "anomaly_quantile must be between 0 and 1."
            )

        if required_votes not in {1, 2}:
            raise ValueError(
                "required_votes must be either 1 or 2."
            )

        self.anomaly_quantile = anomaly_quantile
        self.required_votes = required_votes

        self.thresholds: dict[str, float] = {}
        self.feature_columns: list[str] = []

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        missing = [
            feature
            for feature in self.REQUIRED_FEATURES
            if feature not in x.columns
        ]

        if missing:
            raise ValueError(
                "Rule-based detector is missing required "
                f"features: {missing}"
            )

        benign = x[y.astype(int) == 0]

        if benign.empty:
            raise ValueError(
                "Rule-based detector requires benign training rows."
            )

        total_packets = (
            benign["total_fwd_packets"]
            + benign["total_bwd_packets"]
        )

        self.feature_columns = list(x.columns)

        self.thresholds = {
            "packet_rate": float(
                benign["flow_packets_s"].quantile(
                    self.anomaly_quantile
                )
            ),
            "byte_rate": float(
                benign["flow_bytes_s"].quantile(
                    self.anomaly_quantile
                )
            ),
            "total_packets": float(
                total_packets.quantile(
                    self.anomaly_quantile
                )
            ),
            "forward_packets": float(
                benign["total_fwd_packets"].quantile(
                    self.anomaly_quantile
                )
            ),
            "backward_bytes": float(
                benign["total_bwd_bytes"].quantile(
                    self.anomaly_quantile
                )
            ),
            "backward_packets_low": float(
                benign["total_bwd_packets"].quantile(0.10)
            ),
            "short_duration": float(
                benign["flow_duration"].quantile(0.25)
            ),
        }

    def predict(
        self,
        x: pd.DataFrame,
    ) -> np.ndarray:
        if not self.thresholds:
            raise ValueError(
                "Rule-based detector has not been fitted."
            )

        x = x[self.feature_columns]

        total_packets = (
            x["total_fwd_packets"]
            + x["total_bwd_packets"]
        )

        high_packet_rate = (
            x["flow_packets_s"]
            > self.thresholds["packet_rate"]
        )

        high_byte_rate = (
            x["flow_bytes_s"]
            > self.thresholds["byte_rate"]
        )

        high_rate = (
            high_packet_rate
            | high_byte_rate
        )

        syn_flood = (
            (x["syn_flag_count"] >= 1)
            & (
                x["total_bwd_packets"]
                <= self.thresholds["backward_packets_low"]
            )
            & (
                high_packet_rate
                | (
                    x["total_fwd_packets"]
                    > self.thresholds["forward_packets"]
                )
            )
        )

        short_burst = (
            (
                x["flow_duration"]
                <= self.thresholds["short_duration"]
            )
            & (
                total_packets
                > self.thresholds["total_packets"]
            )
        )

        one_way_volume = (
            (
                x["total_bwd_packets"]
                <= self.thresholds["backward_packets_low"]
            )
            & (
                x["total_fwd_packets"]
                > self.thresholds["forward_packets"]
            )
        )

        amplification = (
            (
                x["total_bwd_bytes"]
                > self.thresholds["backward_bytes"]
            )
            & (
                x["total_bwd_bytes"]
                >= 5 * (x["total_fwd_bytes"] + 1)
            )
        )

        votes = (
            high_rate.astype(int)
            + syn_flood.astype(int)
            + short_burst.astype(int)
            + one_way_volume.astype(int)
            + amplification.astype(int)
        )

        return (
            votes >= self.required_votes
        ).astype(int).to_numpy()