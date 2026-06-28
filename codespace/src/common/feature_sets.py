CIC_COMMON = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "flow_packets_s",
    "flow_bytes_s",
    "syn_flag_count",
    "urg_flag_count",
]

FEATURE_SETS = {
    "all": None,
    "common": CIC_COMMON,
}


def get_feature_set(feature_set_id: str) -> list[str] | None:
    if feature_set_id not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_id}")
    return FEATURE_SETS[feature_set_id]


def resolve_feature_columns(
        available_columns: list[str],
        feature_set_id: str,
        label_column: str,
        drop_columns: list[str] | None = None,
) -> list[str]:
    drop_columns = set(drop_columns or [])
    drop_columns.add(label_column)

    requested = get_feature_set(feature_set_id)

    if requested is None:
        return [c for c in available_columns if c not in drop_columns]

    available = set(available_columns) - drop_columns

    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(f"Missing required features for '{feature_set_id}': {missing}")

    return [c for c in requested if c in available]
