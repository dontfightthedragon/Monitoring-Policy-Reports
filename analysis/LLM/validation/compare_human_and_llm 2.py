# analysis/LLM/validation/compare_pure_vs_key.py
import pandas as pd, numpy as np, re
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# === paths ===
ANN_PURE = "analysis/LLM/validation/manual_labeled_sample_pure.csv"   # activity_id,human_primary_label,human_sdg15_targets
KEY      = "analysis/LLM/validation/manual_validation_key.csv"        # activity_id,model_primary_label,model_sdg15_targets
OUT_DIR  = Path("analysis/LLM/validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_SUMMARY   = OUT_DIR / "manual_validation_summary.csv"
TEX_TABLE     = OUT_DIR / "manual_validation_summary.tex"
CSV_DISAGREE  = OUT_DIR / "manual_validation_disagreements.csv"
CSV_MERGED    = OUT_DIR / "manual_validation_merged.csv"

ALLOWED = {"FORESTS_LAND","WATER_MARINE","WILDLIFE_SPECIES","CROSS_CUTTING","NONE"}

# === helpers ===
def canon_label(x: str) -> str:
    if not isinstance(x, str): return ""
    t = (x.strip().upper()
         .replace("&","AND").replace("/","_").replace("-","_").replace(" ","_"))
    t = re.sub(r"[^A-Z_]", "", t)
    m = {
        "FOREST_LAND":"FORESTS_LAND","FORESTS":"FORESTS_LAND","FOREST":"FORESTS_LAND",
        "WATER":"WATER_MARINE","MARINE":"WATER_MARINE",
        "WILDLIFE":"WILDLIFE_SPECIES","SPECIES":"WILDLIFE_SPECIES",
        "CROSSCUTTING":"CROSS_CUTTING","CROSS__CUTTING":"CROSS_CUTTING",
        "N_A":"NONE","N/A":"NONE","NO_BIODIVERSITY":"NONE","NOT_BIODIVERSITY":"NONE"
    }
    return m.get(t, t)

def norm_targets(s: str) -> str:
    """Accept '15.1,15.5' or '15.1;15.5' or 'SDG15.1 15.3' or 'NONE' -> 'SDG15.x,SDG15.y' (sorted)."""
    if not isinstance(s, str) or not s.strip(): return ""
    s = s.strip().strip('"').strip("'")
    if s.upper() == "NONE": return ""
    toks = re.split(r"[,\s;]+", s)
    out = []
    for t in toks:
        t = t.strip().upper()
        m = re.fullmatch(r"(?:SDG)?15\.(1|2|3|4|5|7|8|9)", t)
        if m: out.append(f"SDG15.{m.group(1)}")
    return ",".join(sorted(set(out)))

def to_set(s: str):
    if not isinstance(s, str) or not s.strip(): return frozenset()
    return frozenset(x.strip() for x in s.split(",") if x.strip())

# === load ===
ann = pd.read_csv(ANN_PURE, dtype=str)
key = pd.read_csv(KEY, dtype=str)

# normalize ids/labels/targets
for df in (ann, key):
    df["activity_id"] = df["activity_id"].astype(str).str.strip()

ann["human_primary_label"]  = ann["human_primary_label"].map(canon_label)
ann["human_sdg15_targets"]  = ann.get("human_sdg15_targets","").map(norm_targets)

# accept aliases in key if present
alias = {
    "primary_label":"model_primary_label",
    "pred_primary_label":"model_primary_label",
    "sdg15_targets":"model_sdg15_targets",
    "pred_sdg15_targets":"model_sdg15_targets",
}
for k,v in alias.items():
    if k in key.columns and "model_primary_label" in v:
        key = key.rename(columns={k:"model_primary_label"})
    if k in key.columns and "model_sdg15_targets" in v:
        key = key.rename(columns={k:"model_sdg15_targets"})

if "model_sdg15_targets" not in key.columns:
    key["model_sdg15_targets"] = ""

key["model_primary_label"] = key["model_primary_label"].map(canon_label)
key["model_sdg15_targets"] = key["model_sdg15_targets"].map(norm_targets)

# === diagnostics ===
missing_from_key = sorted(set(ann["activity_id"]) - set(key["activity_id"]))
missing_from_ann = sorted(set(key["activity_id"]) - set(ann["activity_id"]))
if missing_from_key:
    print(f"[warn] {len(missing_from_key)} ids in manual not in key (dropped on merge)")
if missing_from_ann:
    print(f"[info] {len(missing_from_ann)} ids in key not in manual (unused)")

# === merge ===
df = ann.merge(key, on="activity_id", how="inner", validate="one_to_one")
print(f"Merged n={len(df)}")

# === filter to allowed ===
bad_h = ~df["human_primary_label"].isin(ALLOWED)
bad_m = ~df["model_primary_label"].isin(ALLOWED)
if (bad_h | bad_m).any():
    print(f"[warn] dropping {(bad_h|bad_m).sum()} rows with invalid labels")
    print("Offending human:", sorted(set(df.loc[bad_h, "human_primary_label"])))
    print("Offending model:", sorted(set(df.loc[bad_m, "model_primary_label"])))
    df = df[~(bad_h | bad_m)].copy()

if df.empty:
    raise SystemExit("Empty after filtering — check that manual labels use the fixed taxonomy.")

# === primary metrics ===
y_true = df["human_primary_label"]
y_pred = df["model_primary_label"]
acc   = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

print(f"\nPrimary — Accuracy: {acc:.3f} | Cohen’s κ: {kappa:.3f} | n={len(df)}")
print("\nPer-class precision/recall/F1:")
print(classification_report(y_true, y_pred, labels=sorted(ALLOWED), digits=3))

cm = confusion_matrix(y_true, y_pred, labels=sorted(ALLOWED))
cm_df = pd.DataFrame(cm, index=[f"T:{c}" for c in sorted(ALLOWED)],
                        columns=[f"P:{c}" for c in sorted(ALLOWED)])
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm_df)

# === SDG targets Jaccard ===
hs = df["human_sdg15_targets"].map(to_set)
ms = df["model_sdg15_targets"].map(to_set)
j_all, j_non = [], []
for a,b in zip(hs,ms):
    if len(a)==0 and len(b)==0: j = 1.0
    elif len(a)==0 or len(b)==0: j = 0.0
    else:
        j = len(a & b)/len(a | b); j_non.append(j)
    j_all.append(j)
mean_j = float(np.mean(j_all)) if j_all else np.nan
mean_j_non = float(np.mean(j_non)) if j_non else np.nan
print(f"\nTargets — Mean Jaccard (all): {mean_j:.3f}")
if j_non:
    print(f"Targets — Mean Jaccard (union>0): {mean_j_non:.3f}")

# === exports ===
summary = pd.DataFrame(
    [["Accuracy vs. human", f"{acc:.3f}"],
     ["Cohen’s κ (primary)", f"{kappa:.3f}"],
     ["Mean Jaccard (targets, all)", f"{mean_j:.3f}"],
     ["Mean Jaccard (targets, union>0)", "" if np.isnan(mean_j_non) else f"{mean_j_non:.3f}"]],
    columns=["Metric","Value"]
)
summary.to_csv(CSV_SUMMARY, index=False)

latex = summary.to_latex(index=False, escape=False, column_format="lc",
                         caption=f"Manual validation on n={len(df)} annotated CRS projects.",
                         label="tab:manual_validation")
with open(TEX_TABLE, "w") as f: f.write(latex)

# disagreements
dis = df[df["human_primary_label"] != df["model_primary_label"]].copy()
dis.to_csv(CSV_DISAGREE, index=False)

# merged snapshot (handy for inspection)
df.to_csv(CSV_MERGED, index=False)

print(f"\nSaved: {CSV_SUMMARY}\nSaved: {TEX_TABLE}\nSaved disagreements: {CSV_DISAGREE}\nSaved merged: {CSV_MERGED}")
