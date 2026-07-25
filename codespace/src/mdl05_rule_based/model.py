import numpy as np
import pandas as pd


class RuleBasedDetector:
    CORE_FEATURES = [
        "total_fwd_packets",
        "total_bwd_packets",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "flow_packets_s",
        "flow_bytes_s",
        "syn_flag_count",
    ]

    THRESHOLDS = {
        "high_packet_rate": 25_000.0,
        "extreme_packet_rate": 100_000.0,
        "high_byte_rate": 1_000_000.0,
        "extreme_byte_rate": 10_000_000.0,
        "syn_packet_rate": 10_000.0,
        "one_way_packet_rate": 5_000.0,
        "forward_packets": 20.0,
        "large_packet_count": 40.0,
        "forward_bytes": 10_000.0,
        "amplification_byte_rate": 500_000.0,
        "amplification_ratio": 5.0,
        "fast_iat_mean": 1_000.0,
        "fast_iat_std": 5_000.0,
        "small_packet_mean": 256.0,
        "small_packet_std": 25.0,
        "reset_count": 3.0,
    }

    @staticmethod
    def _column(x: pd.DataFrame, name: str) -> pd.Series:
        if name not in x.columns:
            return pd.Series(0.0, index=x.index)

        return pd.to_numeric(
            x[name],
            errors="coerce",
        ).fillna(0.0)

    @staticmethod
    def _false(x: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=x.index)

    def used_features(self, x: pd.DataFrame) -> list[str]:
        possible = [
            "total_fwd_packets",
            "total_bwd_packets",
            "total_fwd_bytes",
            "total_bwd_bytes",
            "flow_packets_s",
            "flow_bytes_s",
            "syn_flag_count",
            "ack_flag_count",
            "rst_flag_count",
            "psh_flag_count",
            "fin_flag_count",
            "fwd_packets_s",
            "bwd_packets_s",
            "flow_iat_mean",
            "flow_iat_std",
            "packet_length_mean",
            "packet_length_std",
            "fwd_header_length",
            "fwd_act_data_packets",
            "subflow_fwd_packets",
            "subflow_bwd_packets",
            "src_port",
            "dst_port",
        ]

        return [feature for feature in possible if feature in x.columns]

    def _rules(self, x: pd.DataFrame) -> dict[str, pd.Series]:
        missing = [
            feature
            for feature in self.CORE_FEATURES
            if feature not in x.columns
        ]

        if missing:
            raise ValueError(
                "Rule-based detector is missing core features: "
                f"{missing}"
            )

        fwd_packets = self._column(x, "total_fwd_packets")
        bwd_packets = self._column(x, "total_bwd_packets")
        fwd_bytes = self._column(x, "total_fwd_bytes")
        bwd_bytes = self._column(x, "total_bwd_bytes")
        packet_rate = self._column(x, "flow_packets_s")
        byte_rate = self._column(x, "flow_bytes_s")
        syn = self._column(x, "syn_flag_count")

        total_packets = fwd_packets + bwd_packets

        high_rate = (
                (packet_rate >= self.THRESHOLDS["high_packet_rate"])
                | (byte_rate >= self.THRESHOLDS["high_byte_rate"])
        )

        extreme_rate = (
                (packet_rate >= self.THRESHOLDS["extreme_packet_rate"])
                | (byte_rate >= self.THRESHOLDS["extreme_byte_rate"])
        )

        if "ack_flag_count" in x.columns:
            no_ack = self._column(x, "ack_flag_count") == 0
        else:
            no_ack = pd.Series(True, index=x.index)

        syn_flood = (
                (syn >= 1)
                & no_ack
                & (bwd_packets <= 1)
                & (
                        (packet_rate >= self.THRESHOLDS["syn_packet_rate"])
                        | (fwd_packets >= self.THRESHOLDS["forward_packets"])
                )
        )

        one_way_flood = (
                (bwd_packets == 0)
                & (fwd_packets >= self.THRESHOLDS["forward_packets"])
                & (
                        (packet_rate >= self.THRESHOLDS["one_way_packet_rate"])
                        | (fwd_bytes >= self.THRESHOLDS["forward_bytes"])
                )
        )

        amplification = (
                (bwd_packets >= 5)
                & (
                        bwd_bytes
                        >= self.THRESHOLDS["amplification_ratio"]
                        * (fwd_bytes + 1)
                )
                & (
                        byte_rate
                        >= self.THRESHOLDS["amplification_byte_rate"]
                )
        )

        interarrival_burst = self._false(x)

        if {
            "flow_iat_mean",
            "flow_iat_std",
        }.issubset(x.columns):
            interarrival_burst = (
                    (total_packets >= self.THRESHOLDS["large_packet_count"])
                    & (
                            self._column(x, "flow_iat_mean")
                            <= self.THRESHOLDS["fast_iat_mean"]
                    )
                    & (
                            self._column(x, "flow_iat_std")
                            <= self.THRESHOLDS["fast_iat_std"]
                    )
                    & high_rate
            )

        regular_small_packet_burst = self._false(x)

        if {
            "packet_length_mean",
            "packet_length_std",
        }.issubset(x.columns):
            regular_small_packet_burst = (
                    (total_packets >= self.THRESHOLDS["large_packet_count"])
                    & (
                            self._column(x, "packet_length_mean")
                            <= self.THRESHOLDS["small_packet_mean"]
                    )
                    & (
                            self._column(x, "packet_length_std")
                            <= self.THRESHOLDS["small_packet_std"]
                    )
                    & high_rate
            )

        control_packet_flood = self._false(x)

        if {
            "fwd_header_length",
            "fwd_act_data_packets",
        }.issubset(x.columns):
            control_packet_flood = (
                    (fwd_packets >= self.THRESHOLDS["forward_packets"])
                    & (self._column(x, "fwd_act_data_packets") <= 1)
                    & (
                            self._column(x, "fwd_header_length")
                            >= 0.5 * (fwd_bytes + 1)
                    )
                    & high_rate
            )

        reset_flood = self._false(x)

        if "rst_flag_count" in x.columns:
            reset_flood = (
                    (
                            self._column(x, "rst_flag_count")
                            >= self.THRESHOLDS["reset_count"]
                    )
                    & (bwd_packets <= 1)
                    & high_rate
            )

        subflow_burst = self._false(x)

        if {
            "subflow_fwd_packets",
            "subflow_bwd_packets",
        }.issubset(x.columns):
            subflow_burst = (
                    (
                            self._column(x, "subflow_fwd_packets")
                            >= self.THRESHOLDS["large_packet_count"]
                    )
                    & (self._column(x, "subflow_bwd_packets") <= 1)
                    & high_rate
            )

        incomplete_tcp_burst = self._false(x)

        if {
            "ack_flag_count",
            "psh_flag_count",
            "fin_flag_count",
        }.issubset(x.columns):
            incomplete_tcp_burst = (
                    (syn >= 1)
                    & (self._column(x, "ack_flag_count") == 0)
                    & (self._column(x, "psh_flag_count") == 0)
                    & (self._column(x, "fin_flag_count") == 0)
                    & (bwd_packets == 0)
                    & high_rate
            )

        return {
            "extreme_rate": extreme_rate,
            "high_rate": high_rate,
            "syn_flood": syn_flood,
            "one_way_flood": one_way_flood,
            "amplification": amplification,
            "interarrival_burst": interarrival_burst,
            "regular_small_packet_burst": regular_small_packet_burst,
            "control_packet_flood": control_packet_flood,
            "reset_flood": reset_flood,
            "subflow_burst": subflow_burst,
            "incomplete_tcp_burst": incomplete_tcp_burst,
        }

    def predict_with_details(
            self,
            x: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[str, int]]:
        rules = self._rules(x)

        strong_detection = (
                rules["extreme_rate"]
                | rules["syn_flood"]
                | rules["amplification"]
                | rules["incomplete_tcp_burst"]
        )

        supporting_votes = (
                rules["high_rate"].astype(int)
                + rules["one_way_flood"].astype(int)
                + rules["interarrival_burst"].astype(int)
                + rules["regular_small_packet_burst"].astype(int)
                + rules["control_packet_flood"].astype(int)
                + rules["reset_flood"].astype(int)
                + rules["subflow_burst"].astype(int)
        )

        malicious = strong_detection | (supporting_votes >= 2)

        rule_hits = {
            name: int(mask.sum())
            for name, mask in rules.items()
        }
        rule_hits["final_predicted_malicious"] = int(malicious.sum())
        return malicious.astype(int).to_numpy(), rule_hits

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions, _ = self.predict_with_details(x)
        return predictions
