CIC_DEPLOYABLE = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "flow_packets_s",
    "flow_bytes_s",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
]

CIC_COMMON = CIC_DEPLOYABLE

CIC_FULL = CIC_DEPLOYABLE


FEATURE_SETS = {
    "cic_deployable": CIC_DEPLOYABLE,
    "cic_common": CIC_COMMON,
    "cic_full": CIC_FULL,
}


def get_feature_set(feature_set_id: str) -> list[str]:
    if feature_set_id not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_id}")

    return FEATURE_SETS[feature_set_id]


def resolve_feature_columns(
    available_columns: list[str],
    feature_set_id: str,
    label_column: str,
    strict: bool = True,
) -> list[str]:
    if feature_set_id == "all":
        return [col for col in available_columns if col != label_column]

    requested = get_feature_set(feature_set_id)
    available = set(available_columns)

    missing = [col for col in requested if col not in available]

    if missing and strict:
        raise ValueError(
            f"Missing required features for feature set '{feature_set_id}': {missing}"
        )

    return [col for col in requested if col in available]