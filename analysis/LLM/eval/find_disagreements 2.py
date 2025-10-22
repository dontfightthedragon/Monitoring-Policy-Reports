import pandas as pd
import numpy as np

# --- paths ---
gpt35_path = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_3.5.csv"
gpt4o_path = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_4.csv"
text_path  = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_w_text.csv"

# --- load minimal columns robustly ---
gpt35 = pd.read_csv(gpt35_path).rename(columns={"llm_primary_label":"llm_primary_label_gpt35"})
gpt4o = pd.read_csv(gpt4o_path).rename(columns={"llm_primary_label":"llm_primary_label_gpt4o"})
texts = pd.read_csv(text_path)

# merge
df = gpt35.merge(gpt4o, on="activity_id", how="inner")
print(f"Merged rows: {len(df):,}")

# attach text if available (try common column names)
text_cols = [c for c in ["text_for_llm","input_text","llm_text","full_text","text"] if c in texts.columns]
texts_keep = ["activity_id"] + text_cols if text_cols else ["activity_id"]
df = df.merge(texts[texts_keep], on="activity_id", how="left")

# disagreements only
dis = df[df.llm_primary_label_gpt35 != df.llm_primary_label_gpt4o].copy()
print(f"Total disagreements: {len(dis):,}  |  Agreement rate: {(1 - len(dis)/len(df)):.2%}")

# top disagreement pairs (for context)
pair_counts = (
    dis.assign(pair=list(zip(dis.llm_primary_label_gpt35, dis.llm_primary_label_gpt4o)))
       .pair.value_counts()
)
print("\n=== Top disagreement pairs ===")
print(pair_counts.head(12))

# --- helper: pretty-print one row (reuses your text_cols detection) ---
def _print_case(row, max_chars=None):
    print(f"\nActivity {row.activity_id}")
    # print text if present
    txt = None
    for c in text_cols:
        val = row.get(c)
        if pd.notna(val) and str(val).strip():
            txt = str(val); break
    if txt:
        s = txt.strip()
        if max_chars is None:
            print("Text:", s)
        else:
            print("Text:", s[:max_chars] + ("..." if len(s)>max_chars else ""))
    # optional rationales
    if "llm_rationale_gpt35" in dis.columns and pd.notna(row.get("llm_rationale_gpt35")):
        r = str(row["llm_rationale_gpt35"]).strip()
        if r:
            if max_chars is None:
                print("GPT-3.5 rationale:", r)
            else:
                print("GPT-3.5 rationale:", r[:max_chars] + ("..." if len(r)>max_chars else ""))
    if "llm_rationale_gpt4o" in dis.columns and pd.notna(row.get("llm_rationale_gpt4o")):
        r = str(row["llm_rationale_gpt4o"]).strip()
        if r:
            if max_chars is None:
                print("GPT-4o rationale:", r)
            else:
                print("GPT-4o rationale:", r[:max_chars] + ("..." if len(r)>max_chars else ""))
    print("-"*80)

# 1) Random disagreements from anywhere (pair-agnostic)
def sample_any_disagreements(n=5, seed=None, max_chars=None):
    if dis.empty:
        print("No disagreements.")
        return
    k = min(n, len(dis))
    samp = dis.sample(n=k, random_state=seed)
    print(f"\n=== Random disagreements (n={k} of {len(dis)}) ===")
    for _, row in samp.iterrows():
        a = row.llm_primary_label_gpt35
        b = row.llm_primary_label_gpt4o
        print(f"\n--- {a} (GPT-3.5) → {b} (GPT-4o) ---")
        _print_case(row, max_chars=max_chars)

# 2) Random disagreements restricted to a (possibly symmetric) label pair
#    Example: labels=("NONE","CROSS_CUTTING") with symmetric=True
def sample_between(labels=("NONE","CROSS_CUTTING"), n=6, symmetric=True, seed=None, max_chars=None):
    a, b = labels
    if symmetric:
        mask = (
            ((dis.llm_primary_label_gpt35==a) & (dis.llm_primary_label_gpt4o==b)) |
            ((dis.llm_primary_label_gpt35==b) & (dis.llm_primary_label_gpt4o==a))
        )
        subtitle = f"{a} ↔ {b}"
    else:
        mask = (dis.llm_primary_label_gpt35==a) & (dis.llm_primary_label_gpt4o==b)
        subtitle = f"{a} → {b}"
    subset = dis[mask]
    if subset.empty:
        print(f"(no cases) {subtitle}")
        return
    k = min(n, len(subset))
    samp = subset.sample(n=k, random_state=seed)
    # show direction counts for context
    counts = (
        subset.assign(pair=list(zip(subset.llm_primary_label_gpt35, subset.llm_primary_label_gpt4o)))
              .pair.value_counts()
              .to_dict()
    )
    print(f"\n=== Random disagreements within {subtitle} (n={k} of {len(subset)}) ===")
    print("Direction counts:", counts)
    for _, row in samp.iterrows():
        a_ = row.llm_primary_label_gpt35
        b_ = row.llm_primary_label_gpt4o
        print(f"\n--- {a_} (GPT-3.5) → {b_} (GPT-4o) ---")
        _print_case(row, max_chars=max_chars)

# --- summary diagnostics (unchanged) ---
base = df.copy()
base["pair"] = list(zip(base.llm_primary_label_gpt35, base.llm_primary_label_gpt4o))
stay_flip = (base.assign(stay = base.llm_primary_label_gpt35==base.llm_primary_label_gpt4o)
                  .groupby("llm_primary_label_gpt35")["stay"].mean().sort_values(ascending=False))
print("\nStay rate by GPT-3.5 label:\n", (stay_flip*100).round(1).astype(str)+"%")

# Optional: quick keyword probe for one direction (guarded if no text cols)
pair = ("FORESTS_LAND","CROSS_CUTTING")
subset = dis[(dis.llm_primary_label_gpt35==pair[0]) & (dis.llm_primary_label_gpt4o==pair[1])]
if not subset.empty and text_cols:
    txtcol = next((c for c in ["text_for_llm","input_text","text","full_text"] if c in subset.columns), None)
    if txtcol:
        keywords = ["governance","policy","capacity","research","food system","land use","territorial","community","management","planning"]
        counts = {k: subset[txtcol].str.contains(k, case=False, na=False).mean() for k in keywords}
        print("\nKeyword hit rates in {}→{}:".format(*pair), {k: round(v,3) for k,v in counts.items()})

# --- examples (commented) ---
sample_any_disagreements(n=5, seed=0, max_chars=600)
sample_between(("WATER_MARINE","CROSS_CUTTING"), n=6, symmetric=True, seed=0, max_chars=600)   # either direction
# sample_between(("WATER_MARINE","CROSS_CUTTING"), n=6, symmetric=False, seed=42, max_chars=600)  # only NONE → CROSS_CUTTING
