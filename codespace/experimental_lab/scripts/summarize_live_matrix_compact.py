from pathlib import Path
import pandas as pd

RUNS = Path("results/runs_index.csv")
OUT_DIR = Path("results/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "matrix_cic_train_cic_test": "cic_to_cic",
    "matrix_cic_train_botnet_test": "cic_to_botnet",
    "matrix_botnet_train_botnet_test": "botnet_to_botnet",
    "matrix_botnet_train_cic_test": "botnet_to_cic",
}

df = pd.read_csv(RUNS)
df = df[df["dataset_id"].isin(DATASETS.keys())].copy()

if df.empty:
    raise SystemExit("No matrix results found yet.")

df = df.sort_values("run_id")
latest = df.groupby(["model_id", "dataset_id"], as_index=False).tail(1)
latest["direction"] = latest["dataset_id"].map(DATASETS)

f1 = latest.pivot(index="model_id", columns="direction", values="f1_macro")
acc = latest.pivot(index="model_id", columns="direction", values="accuracy")
status = latest.pivot(index="model_id", columns="direction", values="status")

out = pd.DataFrame(index=sorted(latest["model_id"].unique()))
out.index.name = "model_id"

for d in ["cic_to_cic", "cic_to_botnet", "botnet_to_botnet", "botnet_to_cic"]:
    out[f"status_{d}"] = status[d] if d in status.columns else None
    out[f"acc_{d}"] = acc[d] if d in acc.columns else None
    out[f"f1_{d}"] = f1[d] if d in f1.columns else None

out = out.reset_index()

out["f1_drop_cic_to_botnet"] = out["f1_cic_to_cic"] - out["f1_cic_to_botnet"]
out["f1_drop_botnet_to_cic"] = out["f1_botnet_to_botnet"] - out["f1_botnet_to_cic"]

for c in out.columns:
    if c != "model_id" and not c.startswith("status_"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

out = out.sort_values("f1_cic_to_botnet", ascending=False, na_position="last")

terminal_cols = [
    "model_id",
    "f1_cic_to_cic",
    "f1_cic_to_botnet",
    "f1_drop_cic_to_botnet",
    "f1_botnet_to_botnet",
    "f1_botnet_to_cic",
    "f1_drop_botnet_to_cic",
]

compact = out[terminal_cols]

csv_path = OUT_DIR / "live_matrix_compact.csv"
md_path = OUT_DIR / "live_matrix_compact.md"
txt_path = OUT_DIR / "live_matrix_compact.txt"

out.to_csv(csv_path, index=False)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Compact CIC-family transfer matrix\n\n")
    f.write("- `cic_to_cic` = train CIC-DDoS2019 → test CIC-DDoS2019\n")
    f.write("- `cic_to_botnet` = train CIC-DDoS2019 → test botnet-balanced\n")
    f.write("- `botnet_to_botnet` = train botnet-balanced → test botnet-balanced\n")
    f.write("- `botnet_to_cic` = train botnet-balanced → test CIC-DDoS2019\n\n")
    f.write(compact.to_markdown(index=False))
    f.write("\n")

with txt_path.open("w", encoding="utf-8") as f:
    f.write(compact.to_string(index=False))
    f.write("\n")

print("Saved:")
print(" -", csv_path)
print(" -", md_path)
print(" -", txt_path)
print()
print(compact.to_string(index=False))
