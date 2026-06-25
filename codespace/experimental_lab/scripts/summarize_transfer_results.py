import json
from pathlib import Path

import pandas as pd

RUNS_INDEX = Path("results/runs_index.csv")
OUT_DIR = Path("results/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATIVE = "transfer_cicddos2019_native_aligned"
EXTERNAL = "transfer_cicddos2019_to_botnet_balanced_full_strict"

if not RUNS_INDEX.exists():
    raise SystemExit("Missing results/runs_index.csv")

runs = pd.read_csv(RUNS_INDEX)

needed = {"run_id", "model_id", "dataset_id", "status"}
missing = needed - set(runs.columns)
if missing:
    raise SystemExit(f"runs_index.csv is missing columns: {missing}. Columns are: {runs.columns.tolist()}")

runs = runs[runs["dataset_id"].isin([NATIVE, EXTERNAL])].copy()
runs = runs[runs["status"].astype(str).str.lower() == "success"].copy()

if runs.empty:
    raise SystemExit("No successful transfer runs found.")

# Latest successful run per model/dataset.
runs = runs.sort_values("run_id")
runs = runs.groupby(["model_id", "dataset_id"], as_index=False).tail(1)

def read_metric(row, key):
    run_id = row["run_id"]
    p = Path("results/runs") / run_id / "metrics.json"

    if p.exists():
        try:
            data = json.loads(p.read_text())
            if key in data:
                return data[key]
            if "metrics" in data and key in data["metrics"]:
                return data["metrics"][key]
        except Exception:
            pass

    if key in row and pd.notna(row[key]):
        return row[key]

    return None

rows = []
for _, row in runs.iterrows():
    rows.append({
        "model_id": row["model_id"],
        "dataset_id": row["dataset_id"],
        "accuracy": read_metric(row, "accuracy"),
        "f1_macro": read_metric(row, "f1_macro"),
        "run_id": row["run_id"],
    })

df = pd.DataFrame(rows)

wide = df.pivot(index="model_id", columns="dataset_id", values=["accuracy", "f1_macro"])
wide.columns = [f"{metric}__{dataset}" for metric, dataset in wide.columns]
wide = wide.reset_index()

wide = wide.rename(columns={
    f"accuracy__{NATIVE}": "native_accuracy",
    f"f1_macro__{NATIVE}": "native_f1_macro",
    f"accuracy__{EXTERNAL}": "external_accuracy",
    f"f1_macro__{EXTERNAL}": "external_f1_macro",
})

for col in ["native_accuracy", "native_f1_macro", "external_accuracy", "external_f1_macro"]:
    if col not in wide.columns:
        wide[col] = None

wide["accuracy_drop"] = wide["native_accuracy"] - wide["external_accuracy"]
wide["f1_macro_drop"] = wide["native_f1_macro"] - wide["external_f1_macro"]

wide = wide[
    [
        "model_id",
        "native_accuracy",
        "external_accuracy",
        "accuracy_drop",
        "native_f1_macro",
        "external_f1_macro",
        "f1_macro_drop",
    ]
]

wide = wide.sort_values("external_f1_macro", ascending=False, na_position="last")

csv_path = OUT_DIR / "transfer_comparison_summary.csv"
md_path = OUT_DIR / "transfer_comparison_summary.md"

wide.to_csv(csv_path, index=False)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Transfer comparison summary\n\n")
    f.write("Training source: CIC-DDoS2019 aligned train set.\n\n")
    f.write("Native test: CIC-DDoS2019 aligned test set.\n\n")
    f.write("External test: full botnet-balanced CIC-like dataset.\n\n")
    f.write(wide.to_markdown(index=False))
    f.write("\n")

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(wide.to_string(index=False))
