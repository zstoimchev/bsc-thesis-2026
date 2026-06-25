from pathlib import Path
import pandas as pd

RUNS = Path("results/runs_index.csv")
OUT_DIR = Path("results/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "matrix_cic_train_cic_test",
    "matrix_cic_train_botnet_test",
    "matrix_botnet_train_botnet_test",
    "matrix_botnet_train_cic_test",
]

RENAMES = {
    "matrix_cic_train_cic_test": "original_to_original",
    "matrix_cic_train_botnet_test": "original_to_other",
    "matrix_botnet_train_botnet_test": "other_to_other",
    "matrix_botnet_train_cic_test": "other_to_original",
}

if not RUNS.exists():
    raise SystemExit("Missing results/runs_index.csv")

df = pd.read_csv(RUNS)
df = df[df["dataset_id"].isin(DATASETS)].copy()

if df.empty:
    raise SystemExit("No matrix results found yet.")

df = df.sort_values("run_id")
latest = df.groupby(["model_id", "dataset_id"], as_index=False).tail(1)

rows = []
for _, r in latest.iterrows():
    short = RENAMES[r["dataset_id"]]
    rows.append({
        "model_id": r["model_id"],
        "direction": short,
        "status": r.get("status"),
        "accuracy": r.get("accuracy"),
        "balanced_accuracy": r.get("balanced_accuracy"),
        "f1_macro": r.get("f1_macro"),
        "f1_weighted": r.get("f1_weighted"),
        "roc_auc": r.get("roc_auc"),
        "duration_seconds": r.get("duration_seconds"),
        "run_id": r.get("run_id"),
    })

long = pd.DataFrame(rows)

# Wide metric table
wide_parts = []

for metric in ["status", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc"]:
    p = long.pivot(index="model_id", columns="direction", values=metric)
    p.columns = [f"{metric}_{c}" for c in p.columns]
    wide_parts.append(p)

wide = pd.concat(wide_parts, axis=1).reset_index()

# Add drops if columns exist
def add_drop(new_col, a, b):
    if a in wide.columns and b in wide.columns:
        wide[new_col] = pd.to_numeric(wide[a], errors="coerce") - pd.to_numeric(wide[b], errors="coerce")

add_drop("accuracy_drop_original_to_other", "accuracy_original_to_original", "accuracy_original_to_other")
add_drop("f1_drop_original_to_other", "f1_macro_original_to_original", "f1_macro_original_to_other")
add_drop("accuracy_drop_other_to_original", "accuracy_other_to_other", "accuracy_other_to_original")
add_drop("f1_drop_other_to_original", "f1_macro_other_to_other", "f1_macro_other_to_original")

# Useful ordering
sort_col = "f1_macro_original_to_other"
if sort_col in wide.columns:
    wide = wide.sort_values(sort_col, ascending=False, na_position="last")

csv_path = OUT_DIR / "live_matrix_summary.csv"
md_path = OUT_DIR / "live_matrix_summary.md"
status_path = OUT_DIR / "live_matrix_status_counts.csv"

wide.to_csv(csv_path, index=False)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Live CIC-family transfer matrix summary\n\n")
    f.write("Directions:\n\n")
    f.write("- original_to_original = CIC-DDoS2019 train → CIC-DDoS2019 test\n")
    f.write("- original_to_other = CIC-DDoS2019 train → botnet-balanced test\n")
    f.write("- other_to_other = botnet-balanced train → botnet-balanced test\n")
    f.write("- other_to_original = botnet-balanced train → CIC-DDoS2019 test\n\n")
    f.write(wide.to_markdown(index=False))
    f.write("\n")

counts = (
    long.groupby(["direction", "status"])
    .size()
    .reset_index(name="count")
    .sort_values(["direction", "status"])
)
counts.to_csv(status_path, index=False)

print("Saved:", csv_path)
print("Saved:", md_path)
print("Saved:", status_path)
print()
print("Status counts:")
print(counts.to_string(index=False))
print()
print("Current summary:")
print(wide.to_string(index=False))
