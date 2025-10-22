# analysis/LLM/validation/compare_gpt4o_vs_gpt35.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- CONFIG ---
A_CSV = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_4.csv"   # main model
B_CSV = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_3.5.csv"   # validation model
A_COL = "llm_primary_label"                             # ensure column name matches your file
B_COL = "llm_primary_label"
IDCOL = "activity_id"
OUT   = "analysis/LLM/figures"; os.makedirs(OUT, exist_ok=True)

# consistent order for plotting
ORDER = ["FORESTS_LAND","WATER_MARINE","WILDLIFE_SPECIES","CROSS_CUTTING","NONE"]

# --- LOAD ---
g4 = pd.read_csv(A_CSV, usecols=[IDCOL, A_COL]).rename(columns={A_COL:"gpt4o"})
g3 = pd.read_csv(B_CSV, usecols=[IDCOL, B_COL]).rename(columns={B_COL:"gpt35"})
df = g4.merge(g3, on=IDCOL, how="inner").dropna()

# ensure both columns follow same category order
for col in ["gpt4o","gpt35"]:
    df[col] = pd.Categorical(df[col].where(df[col].isin(ORDER), "NONE"), categories=ORDER)

# --- 1) LABEL DISTRIBUTION COMPARISON ---
def label_share(series, order):
    c = series.value_counts().reindex(order, fill_value=0)
    return c / c.sum() * 100

share4 = label_share(df["gpt4o"], ORDER)
share3 = label_share(df["gpt35"], ORDER)

x = np.arange(len(ORDER))
w = 0.38
fig, ax = plt.subplots(figsize=(8,4.2))
ax.bar(x - w/2, share4.values, width=w, label="GPT-4o", alpha=0.85)
ax.bar(x + w/2, share3.values, width=w, label="GPT-3.5", alpha=0.85)
for xi, v in zip(x - w/2, share4.values):
    ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w/2, share3.values):
    ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=20, ha="right")
ax.set_ylabel("Share of labels (%)")
ax.set_title("Label distribution comparison: GPT-4o vs GPT-3.5")
ax.legend()
plt.tight_layout()
dist_path = f"{OUT}/gpt4o_vs_gpt35_label_distribution.pdf"
plt.savefig(dist_path)

# --- 2) CONFUSION HEATMAP ---
cm = confusion_matrix(df["gpt4o"], df["gpt35"], labels=ORDER)
cm_pct = cm / cm.sum(axis=1, keepdims=True)
cm_pct = np.nan_to_num(cm_pct)

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(ORDER)))
ax.set_yticks(np.arange(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=30, ha="right")
ax.set_yticklabels(ORDER)
ax.set_xlabel("GPT-3.5 prediction")
ax.set_ylabel("GPT-4o prediction")
ax.set_title("Confusion heatmap: GPT-4o vs GPT-3.5 (row %)")
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        ax.text(j, i, f"{cm_pct[i,j]*100:.0f}%", ha="center", va="center", fontsize=8)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Row-normalized share (%)")
plt.tight_layout()
cm_path = f"{OUT}/gpt4o_vs_gpt35_confusion_heatmap.pdf"
plt.savefig(cm_path)

print(f"Saved: \n {dist_path}\n {cm_path}")
print(f"Compared rows: {len(df):,}")
