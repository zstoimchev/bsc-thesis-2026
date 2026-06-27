import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_packets",
    "total_bytes",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "flow_bytes_s",
    "flow_packets_s",
    "syn_flag_count",
    "urg_flag_count",
]


def generate_benign_flows(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    flow_duration = rng.uniform(1.0, 120.0, rows)

    total_fwd_packets = rng.poisson(lam=8, size=rows) + 1
    total_bwd_packets = rng.poisson(lam=6, size=rows) + 1

    fwd_packet_length_mean = rng.uniform(60.0, 900.0, rows)
    bwd_packet_length_mean = rng.uniform(60.0, 900.0, rows)

    total_fwd_bytes = total_fwd_packets * fwd_packet_length_mean
    total_bwd_bytes = total_bwd_packets * bwd_packet_length_mean

    total_packets = total_fwd_packets + total_bwd_packets
    total_bytes = total_fwd_bytes + total_bwd_bytes

    flow_bytes_s = total_bytes / flow_duration
    flow_packets_s = total_packets / flow_duration

    syn_flag_count = rng.binomial(n=2, p=0.15, size=rows)
    urg_flag_count = rng.binomial(n=1, p=0.05, size=rows)

    ack_flag_count = rng.binomial(n=total_packets, p=0.75)
    rst_flag_count = rng.binomial(n=1, p=0.03, size=rows)

    return pd.DataFrame(
        {
            "flow_duration": flow_duration,
            "total_fwd_packets": total_fwd_packets,
            "total_bwd_packets": total_bwd_packets,
            "total_fwd_bytes": total_fwd_bytes,
            "total_bwd_bytes": total_bwd_bytes,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "fwd_packet_length_mean": fwd_packet_length_mean,
            "bwd_packet_length_mean": bwd_packet_length_mean,
            "flow_bytes_s": flow_bytes_s,
            "flow_packets_s": flow_packets_s,
            "syn_flag_count": syn_flag_count,
            "urg_flag_count": urg_flag_count,
            "label": "Benign",
            "ack_flag_count": ack_flag_count,
            "rst_flag_count": rst_flag_count,
        }
    )


def generate_attack_flows(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    flow_duration = rng.uniform(0.001, 10.0, rows)

    total_fwd_packets = rng.poisson(lam=80, size=rows) + 1
    total_bwd_packets = rng.poisson(lam=2, size=rows)

    fwd_packet_length_mean = rng.uniform(40.0, 300.0, rows)
    bwd_packet_length_mean = rng.uniform(20.0, 250.0, rows)

    total_fwd_bytes = total_fwd_packets * fwd_packet_length_mean
    total_bwd_bytes = total_bwd_packets * bwd_packet_length_mean

    total_packets = total_fwd_packets + total_bwd_packets
    total_bytes = total_fwd_bytes + total_bwd_bytes

    safe_duration = np.clip(flow_duration, 1e-6, None)

    flow_bytes_s = total_bytes / safe_duration
    flow_packets_s = total_packets / safe_duration

    syn_flag_count = rng.poisson(lam=25, size=rows)
    urg_flag_count = rng.binomial(n=2, p=0.10, size=rows)

    ack_flag_count = rng.binomial(n=np.maximum(total_packets, 1), p=0.30)
    rst_flag_count = rng.poisson(lam=3, size=rows)

    return pd.DataFrame(
        {
            "flow_duration": flow_duration,
            "total_fwd_packets": total_fwd_packets,
            "total_bwd_packets": total_bwd_packets,
            "total_fwd_bytes": total_fwd_bytes,
            "total_bwd_bytes": total_bwd_bytes,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "fwd_packet_length_mean": fwd_packet_length_mean,
            "bwd_packet_length_mean": bwd_packet_length_mean,
            "flow_bytes_s": flow_bytes_s,
            "flow_packets_s": flow_packets_s,
            "syn_flag_count": syn_flag_count,
            "urg_flag_count": urg_flag_count,
            "label": "Attack",
            "ack_flag_count": ack_flag_count,
            "rst_flag_count": rst_flag_count,
        }
    )


def generate_dummy_dataset(
    rows: int,
    attack_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attack_rows = int(rows * attack_ratio)
    benign_rows = rows - attack_rows

    benign_df = generate_benign_flows(benign_rows, rng)
    attack_df = generate_attack_flows(attack_rows, rng)

    df = pd.concat([benign_df, attack_df], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate dummy IDS flow dataset for testing the thesis pipeline."
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to generate.",
    )

    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.30,
        help="Ratio of attack rows. Example: 0.30 means 30 percent attacks.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/dummy.csv",
        help="Output CSV path relative to the codespace root.",
    )

    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0.")

    if not 0.0 < args.attack_ratio < 1.0:
        raise ValueError("--attack-ratio must be between 0 and 1.")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dummy_dataset(
        rows=args.rows,
        attack_ratio=args.attack_ratio,
        seed=args.seed,
    )

    df.to_csv(output_path, index=False)

    print(f"Saved dummy dataset to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Label distribution:")
    print(df["label"].value_counts())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())