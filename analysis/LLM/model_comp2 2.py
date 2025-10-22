import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

#  File paths 
gpt35_path = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_3.5.csv"
gpt4o_path = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_4.csv"
OUT_TEX    = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/figures/table_gpt4o_vs_gpt35.tex"

PRIMARY_COL = "llm_primary_label"
TARGETS_COL = "llm_sdg15_targets"

# Load and merge datasets 
g35 = pd.read_csv(gpt35_path, usecols=["activity_id", PRIMARY_COL, TARGETS_COL], dtype=str)
g4  = pd.read_csv(gpt4o_path,  usecols=["activity_id", PRIMARY_COL, TARGETS_COL], dtype=str)
g   = g4.merge(g35, on="activity_id", how="inner", suffixes=("_gpt4o","_gpt35")).dropna(
    subset=[PRIMARY_COL+"_gpt4o", PRIMARY_COL+"_gpt35"]
)
print(f"Merged {len(g):,} activities.")

#  Label shares 
labels = ["CROSS_CUTTING","FORESTS_LAND","NONE","WATER_MARINE","WILDLIFE_SPECIES"]
def shares(s): 
    return (s.value_counts(normalize=True) * 100).reindex(labels).fillna(0)

share_4  = shares(g[PRIMARY_COL+"_gpt4o"])
share_35 = shares(g[PRIMARY_COL+"_gpt35"])
diff_pp  = (share_4 - share_35)

#  Agreement and Kappa 
agree_rate = (g[PRIMARY_COL+"_gpt4o"] == g[PRIMARY_COL+"_gpt35"]).mean()
kappa = cohen_kappa_score(g[PRIMARY_COL+"_gpt4o"], g[PRIMARY_COL+"_gpt35"])

#  Mean Jaccard similarity for SDG targets
def parse_targets(x):
    if pd.isna(x) or str(x).strip()=="":
        return frozenset()
    return frozenset(t.strip() for t in str(x).split(",") if t.strip())

t4  = g[TARGETS_COL+"_gpt4o"].map(parse_targets)
t35 = g[TARGETS_COL+"_gpt35"].map(parse_targets)

jaccard_scores = []
for a, b in zip(t4, t35):
    if len(a) == 0 and len(b) == 0:
        j = 1.0
    elif len(a) == 0 or len(b) == 0:
        j = 0.0
    else:
        j = len(a & b) / len(a | b)
    jaccard_scores.append(j)

mean_jaccard = np.mean(jaccard_scores)

mask_union = [(len(a|b) > 0) for a,b in zip(t4, t35)]
mean_jaccard_nonempty = np.mean([j for j,m in zip(jaccard_scores, mask_union) if m]) if any(mask_union) else np.nan

#  Build final table 
rows = pd.DataFrame({
    "GPT-4o (%)": share_4.round(1),
    "GPT-3.5 (%)": share_35.round(1),
    "Difference (pp)": diff_pp.round(1)
})
rows.index.name = "Primary label"

summ = pd.DataFrame(index=[
    "Agreement (primary label)",
    "Cohen’s κ (primary label)",
    "Mean Jaccard (targets, all rows)",
    "Mean Jaccard (targets, union>0)"
], columns=rows.columns).astype("object")

summ.loc["Agreement (primary label)", "GPT-4o (%)"] = f"{agree_rate*100:.1f}%"
summ.loc["Cohen’s κ (primary label)", "GPT-4o (%)"] = f"{kappa:.3f}"
summ.loc["Mean Jaccard (targets, all rows)", "GPT-4o (%)"] = f"{mean_jaccard:.3f}"
summ.loc["Mean Jaccard (targets, union>0)", "GPT-4o (%)"] = f"{mean_jaccard_nonempty:.3f}"

final_tbl = pd.concat([rows, summ])

final_tbl = final_tbl.fillna("")  # hide NaNs in summary rows

latex = final_tbl.to_latex(
    escape=False,
    bold_rows=False,
    column_format="lccc",
    caption=f"Comparison of GPT-4o and GPT-3.5 on n={len(g):,} activities.",
    label="tab:gpt4o_vs_gpt35"
)
with open(OUT_TEX, "w") as f:
    f.write(latex)

# optional sanity checks 
print(f"\nSum GPT-4o = {share_4.sum():.1f}%,  Sum GPT-3.5 = {share_35.sum():.1f}%")


#  Export LaTeX 
latex = final_tbl.to_latex(
    escape=False,
    bold_rows=False,
    column_format="lccc",
    caption=f"Comparison of GPT-4o and GPT-3.5 classifications on n={len(g):,} activities.",
    label="tab:gpt4o_vs_gpt35"
)
with open(OUT_TEX, "w") as f:
    f.write(latex)

print(final_tbl)
print(f"\nSaved LaTeX to: {OUT_TEX}")
