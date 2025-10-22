import pandas as pd
import numpy as np

# Inputs
PRED = "analysis/LLM/artifacts/full/llm_predictions_pp_4.csv"
CRS  = "analysis/LLM/artifacts/full/inference_input.csv"
OUT_ANN = "analysis/LLM/validation/manual_validation_sample_blind.csv"   # annotate this (no model cols)
OUT_KEY = "analysis/LLM/validation/manual_validation_key.csv"            # keep hidden (model cols)

# Load
pred = pd.read_csv(PRED, dtype=str)
crs  = pd.read_csv(CRS, dtype=str)

# Choose text column from input
text_priority = [
    "text_for_llm","text_for_llm_en","full_text_en","full_text",
    "long_description_en","long_description","title_en","title",
    "short_description_en","short_description"
]
text_cols = [c for c in text_priority if c in crs.columns]
if not text_cols:
    raise ValueError("No suitable text column in inference_input.csv")
cols_keep = ["activity_id"] + text_cols

# Merge labels ↔ text (labels used only for sampling; we'll hide them in the blind file)
# Note: CRS has duplicate activity_ids (with suffixes), so this is many-to-many
df = pred.merge(crs[cols_keep], on="activity_id", how="left")

# Pick a single display text
def pick_text(row):
    for c in text_cols:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
df["orig_text"] = df.apply(pick_text, axis=1)

# (Optional) de-duplicate identical texts so you don’t review repeats
df["text_hash"] = df["orig_text"].fillna("").str.strip().map(lambda s: pd.util.hash_pandas_object(pd.Series([s])).iloc[0])
dfu = df.drop_duplicates("text_hash").copy()

# Stratified sample across predicted labels (but DO NOT include those labels in the blind file)
np.random.seed(42)
n_total = 100
labels = dfu["llm_primary_label"].dropna().unique().tolist()
per_lab = max(1, n_total // max(1, len(labels)))

parts = []
for lab in labels:
    part = dfu[dfu["llm_primary_label"]==lab].sample(min(per_lab, len(dfu[dfu["llm_primary_label"]==lab])), random_state=42)
    parts.append(part)
sample = pd.concat(parts).drop_duplicates("activity_id")
if len(sample) < n_total:
    sample = pd.concat([sample, dfu[~dfu["activity_id"].isin(sample["activity_id"])].sample(n_total-len(sample), random_state=42)])

sample = sample.head(n_total).copy()

# --- BLIND FILE (for you to annotate; no model columns) ---
blind = sample[["activity_id","orig_text"]].copy()
blind["human_primary_label"] = ""      # fill with {FORESTS_LAND,WATER_MARINE,WILDLIFE_SPECIES,CROSS_CUTTING,NONE}
blind["human_sdg15_targets"] = ""      # optional, comma-separated (e.g., SDG15.1,SDG15.2)
blind.to_csv(OUT_ANN, index=False)
print("Wrote blind file for annotation:", OUT_ANN)

# --- KEY FILE (keep hidden; used later for scoring) ---
key = sample[["activity_id","llm_primary_label","llm_sdg15_targets"]].rename(columns={
    "llm_primary_label":"model_primary_label",
    "llm_sdg15_targets":"model_sdg15_targets"
})
key.to_csv(OUT_KEY, index=False)
print("Wrote key file for scoring:", OUT_KEY)
