from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images"
OUT_DIR.mkdir(exist_ok=True)

model_metrics = [
    {"Model": "Altman Z'-Score", "ROC-AUC": 0.7602, "F1-Score": 0.0668},
    {"Model": "Logistic Regression", "ROC-AUC": 0.7216, "F1-Score": 0.0659},
    {"Model": "Decision Tree", "ROC-AUC": 0.7877, "F1-Score": 0.0508},
    {"Model": "PySR (random_state=0)", "ROC-AUC": 0.8000, "F1-Score": 0.0621},
    {"Model": "PySR (random_state=42)", "ROC-AUC": 0.7697, "F1-Score": 0.0633},
]
model_df = pd.DataFrame(model_metrics).sort_values("ROC-AUC", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4C72B0" if model != "PySR (random_state=0)" else "#55A868" for model in model_df["Model"]]
positions = range(len(model_df))
ax.bar(positions, model_df["ROC-AUC"], color=colors, edgecolor="black")
ax.set_title("Test ROC-AUC by model", fontsize=14, weight="bold")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0, 0.85)
ax.set_xticks(list(positions))
ax.set_xticklabels(model_df["Model"], rotation=25, ha="right")
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.4f}" for v in container.datavalues], padding=3, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "model_roc_auc.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
positions = range(len(model_df))
ax.bar(positions, model_df["F1-Score"], color=["#4C72B0" if model != "PySR (random_state=0)" else "#55A868" for model in model_df["Model"]], edgecolor="black")
ax.set_title("Test F1-score by model", fontsize=14, weight="bold")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 0.08)
ax.set_xticks(list(positions))
ax.set_xticklabels(model_df["Model"], rotation=25, ha="right")
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.4f}" for v in container.datavalues], padding=3, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "model_f1_score.png", dpi=300)
plt.close(fig)

temporal = [
    {"Time Period": "Train (Early) 2008-2015", "ROC-AUC": 0.7009, "F1-Score": 0.0294, "Bankruptcy Rate": 0.0067},
    {"Time Period": "Validate (Mid) 2016-2017", "ROC-AUC": 0.7038, "F1-Score": 0.0377, "Bankruptcy Rate": 0.0087},
    {"Time Period": "Test (Late) 2018-2019", "ROC-AUC": 0.7193, "F1-Score": 0.0476, "Bankruptcy Rate": 0.0104},
    {"Time Period": "COVID 2020-2021", "ROC-AUC": 0.7626, "F1-Score": 0.0239, "Bankruptcy Rate": 0.0042},
    {"Time Period": "Post-COVID 2022-2023", "ROC-AUC": 0.6734, "F1-Score": 0.0451, "Bankruptcy Rate": 0.0116},
]
temporal_df = pd.DataFrame(temporal)

fig, ax = plt.subplots(figsize=(11, 6))
positions = range(len(temporal_df))
ax.plot(positions, temporal_df["ROC-AUC"], marker="o", linewidth=2.5, color="#C44E52", markersize=8)
ax.set_title("Temporal ROC-AUC of the early-period symbolic-regression score", fontsize=14, weight="bold")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.60, 0.80)
ax.set_xticks(list(positions))
ax.set_xticklabels(temporal_df["Time Period"], rotation=25, ha="right")
for x, y in zip(positions, temporal_df["ROC-AUC"]):
    ax.text(x, y + 0.005, f"{y:.4f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "temporal_roc_auc.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))
positions = range(len(temporal_df))
ax.bar(positions, temporal_df["Bankruptcy Rate"], color="#7A6FBE", edgecolor="black")
ax.set_title("Bankruptcy rate by time period", fontsize=14, weight="bold")
ax.set_ylabel("Bankruptcy rate")
ax.set_ylim(0, 0.015)
ax.set_xticks(list(positions))
ax.set_xticklabels(temporal_df["Time Period"], rotation=25, ha="right")
for container in ax.containers:
    ax.bar_label(container, labels=[f"{v:.4f}" for v in container.datavalues], padding=3, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "bankruptcy_rate_by_period.png", dpi=300)
plt.close(fig)

print(f"Saved visuals to {OUT_DIR}")
for path in sorted(OUT_DIR.glob("*.png")):
    print(path.name)
