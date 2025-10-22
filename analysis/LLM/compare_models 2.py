import pandas as pd, json, numpy as np, re
from collections import Counter
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score

# --- INPUT PATHS ---
gpt35_csv = Path("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_3.5.csv")
gpt4o_csv = Path("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_4.csv")

assert gpt35_csv.exists(), f"GPT-3.5 not found: {gpt35_csv}"
assert gpt4o_csv.exists(), f"GPT-4o not found: {gpt4o_csv}"

print("GPT-3.5 raw label uniques (head):",
      pd.read_csv(gpt35_csv).iloc[:1000]["llm_primary_label"].astype(str).str.upper().str.slice(0,40).unique()[:20])
print("GPT-4o raw label uniques (head):",
      pd.read_csv(gpt4o_csv).iloc[:1000]["llm_primary_label"].astype(str).str.upper().str.slice(0,40).unique()[:20])

# --- Canonical sets ---
ALLOWED_PL = {"CROSS_CUTTING","FORESTS_LAND","NONE","WATER_MARINE","WILDLIFE_SPECIES"}
ALLOWED_TG = {f"SDG15.{i}" for i in [1,2,3,4,5,7,8,9]}

# --- Helpers ---
def _autocols(cols):
    lower = {c.lower(): c for c in cols}
    for k in ["activity_id","project_id","id"]:
        if k in lower: id_col = lower[k]; break
    else: raise ValueError(f"ID column not found in {cols}")
    for k in ["llm_primary_label","primary_label","pl","label"]:
        if k in lower: pl_col = lower[k]; break
    else: raise ValueError(f"primary_label column not found in {cols}")
    for k in ["llm_sdg15_targets","sdg15_targets","sdg15_targets_json","targets","tg"]:
        if k in lower: tg_col = lower[k]; break
    else: raise ValueError(f"targets column not found in {cols}")
    return id_col, pl_col, tg_col

def _parse_targets(x):
    if x is None or (isinstance(x,float) and pd.isna(x)) or x=="":
        return []
    if isinstance(x,list):
        cand = [str(t).strip() for t in x]
    else:
        s = str(x).strip()
        try:
            j = json.loads(s)
            cand = [str(t).strip() for t in j] if isinstance(j,list) else [s]
        except Exception:
            cand = [t.strip() for t in s.split(",") if t.strip()]
    norm = []
    for t in cand:
        t = t.upper().replace("SDG 15.","SDG15.")
        m = re.match(r'^(?:SDG15\.)?0?([1-9])$', t)
        if m:
            t = f"SDG15.{m.group(1)}"
            if t in ALLOWED_TG:
                norm.append(t)
    return sorted(set(norm))

def _canon_pl(x):
    s = str(x).strip().upper()
    s = re.sub(r'[^A-Z0-9]+','_',s); s = re.sub(r'_+','_',s).strip('_')
    mapping = {
        "WILDLIFE":"WILDLIFE_SPECIES",
        "WILDLIFE_SPECIES":"WILDLIFE_SPECIES",
        "FORESTS_LAND":"FORESTS_LAND",
        "FORESTRY_LAND":"FORESTS_LAND",
        "WATER_MARINE":"WATER_MARINE",
        "CROSSCUTTING":"CROSS_CUTTING",
        "CROSS_CUTTING":"CROSS_CUTTING",
        "NONE":"NONE",
    }
    out = mapping.get(s, s)
    return out if out in ALLOWED_PL else "NONE"

def load_normalized(fp):
    df = pd.read_csv(fp)
    id_col, pl_col, tg_col = _autocols(df.columns)
    df = df[[id_col, pl_col, tg_col]].dropna(subset=[id_col]).copy()
    df[id_col] = df[id_col].astype(str)
    df = df.drop_duplicates(subset=[id_col], keep="first")
    df["id"] = df[id_col].astype(str)
    df["pl"] = df[pl_col].apply(_canon_pl)
    df["tg"] = df[tg_col].apply(_parse_targets)
    empty_tg = (df["tg"].str.len()==0).mean()
    none_rate = (df["pl"]=="NONE").mean()
    print(f"[{fp.name}] rows={len(df)} | NONE_share={none_rate:.3f} | empty_targets={empty_tg:.3f}")
    return df[["id","pl","tg"]]

# --- Load ---
g35 = load_normalized(gpt35_csv)
g4  = load_normalized(gpt4o_csv)

# --- Align on ID ---
m = g35.merge(g4, on="id", suffixes=("_gpt35","_gpt4o"), how="inner")
print({"gpt35_rows":len(g35), "gpt4o_rows":len(g4), "overlap":len(m)})

# --- Agreement metrics (primary label) ---
agree = (m["pl_gpt35"] == m["pl_gpt4o"]).mean()
kappa = cohen_kappa_score(m["pl_gpt35"], m["pl_gpt4o"])

# --- Distributional divergence (JSD) ---
def dist(series):
    c = Counter(series)
    labs = sorted(c)
    p = np.array([c[L] for L in labs], dtype=float); p /= p.sum()
    return labs, p

L_35, p_35 = dist(m["pl_gpt35"])
L_4,  p_4  = dist(m["pl_gpt4o"])
L_all = sorted(set(L_35) | set(L_4))
def vec(Lref, Lsrc, psrc):
    idx = {l:i for i,l in enumerate(Lsrc)}
    v = np.array([psrc[idx[l]] if l in idx else 0.0 for l in Lref], dtype=float)
    s = v.sum()
    return v if s==0 else v/s

v35, v4 = vec(L_all, L_35, p_35), vec(L_all, L_4, p_4)
jsd = float(jensenshannon(v35, v4))

# --- Per-label share deltas (GPT-3.5 minus GPT-4o) ---
share35 = m["pl_gpt35"].value_counts(normalize=True)
share4  = m["pl_gpt4o"].value_counts(normalize=True)
per_label_delta = (share35 - share4).reindex(L_all).fillna(0).round(4).to_dict()

# --- Targets: mean Jaccard ---
def jaccard(a,b):
    a,b = set(a), set(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

mean_jacc = float(np.mean([jaccard(a,b) for a,b in zip(m["tg_gpt35"], m["tg_gpt4o"])]))

print({
    "n_overlap": len(m),
    "primary_agreement_pct": round(agree,4),
    "cohens_kappa": round(kappa,4),
    "js_divergence_primary": round(jsd,4),
    "per_label_delta_share(gpt35_minus_gpt4o)": per_label_delta,
    "mean_jaccard_targets": round(mean_jacc,4)
})

row = {
    "n_overlap": 39082,
    "agreement": 0.6463,
    "cohen_kappa": 0.5615,
    "JSD_primary": 0.1905,
    "mean_Jaccard_targets": 0.7147,
    "Δshare_FORESTS_LAND(3.5-4o)": 0.0943,
    "Δshare_WILDLIFE_SPECIES": 0.0995,
    "Δshare_WATER_MARINE": 0.0119,
    "Δshare_CROSS_CUTTING": -0.1674,
    "Δshare_NONE": -0.0384
}
tbl = pd.DataFrame([row])
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
tbl.to_csv("analysis/LLM/eval/gpt35_vs_gpt4o_summary.csv", index=False)

OUT_TEX    = "analysis/LLM/figures/table_gpt4o_vs_gpt35.tex"
PRIMARY_COL = "llm_primary_label"            # required
TARGETS_COL = "llm_sdg15_targets"             # optional

# --- load & align ---
g35 = pd.read_csv(gpt35_csv, usecols=["activity_id", PRIMARY_COL, TARGETS_COL], dtype=str)
g4  = pd.read_csv(gpt4o_csv,  usecols=["activity_id", PRIMARY_COL, TARGETS_COL], dtype=str)
g   = g4.merge(g35, on="activity_id", how="inner", suffixes=("_gpt4o","_gpt35")).dropna(subset=[PRIMARY_COL+"_gpt4o", PRIMARY_COL+"_gpt35"])

# --- label shares ---
labels = ["CROSS_CUTTING","FORESTS_LAND","NONE","WATER_MARINE","WILDLIFE_SPECIES"]
def shares(s): 
    return (s.value_counts(normalize=True) * 100).reindex(labels).fillna(0)

share_4  = shares(g[PRIMARY_COL+"_gpt4o"])
share_35 = shares(g[PRIMARY_COL+"_gpt35"])
diff_pp  = (share_4 - share_35)

tbl = pd.DataFrame({
    "GPT-4o-mini (%)": share_4.round(1),
    "GPT-3.5-turbo (%)": share_35.round(1),
    "Difference (pp)": diff_pp.round(1)
})
tbl.index.name = "Primary label"

#  agreement (primary) 
agree_rate = (g[PRIMARY_COL+"_gpt4o"] == g[PRIMARY_COL+"_gpt35"]).mean()

#  mean Jaccard over SDG15 targets (optional) 
def parse_targets(x):
    if pd.isna(x) or str(x).strip()=="":
        return frozenset()
    return frozenset(t.strip() for t in str(x).split(",") if t.strip())

if TARGETS_COL in g.columns:
    t4  = g[TARGETS_COL+"_gpt4o"].map(parse_targets)
    t35 = g[TARGETS_COL+"_gpt35"].map(parse_targets)
    inter = (t4 & t35).map(len)
    union = (t4 | t35).map(len).replace(0, np.nan)
    mean_jaccard = (inter/union).fillna(1.0).mean()  
else:
    mean_jaccard = np.nan

# append summary rows (like your screenshot) 
summary = pd.DataFrame({
    "GPT-4o-mini (%)": ["", ""],
    "GPT-3.5-turbo (%)": ["", ""],
    "Difference (pp)": ["", ""]
}, index=["Agreement (primary label)", "Mean Jaccard (targets)"])

# pretty strings
summary.loc["Agreement (primary label)","GPT-4o-mini (%)"] = f"{agree_rate*100:.1f}%"
summary.loc["Mean Jaccard (targets)","GPT-4o-mini (%)"]   = (f"{mean_jaccard:.3g}" if pd.notna(mean_jaccard) else "—")

final_tbl = pd.concat([tbl, summary])

# --- export LaTeX ---
latex = final_tbl.to_latex(escape=False, bold_rows=False, column_format="lccc")
with open(OUT_TEX, "w") as f:
    f.write(latex)

print(final_tbl)
print(f"\nSaved LaTeX to: {OUT_TEX}")
