# analysis/LLM/eval/normalize_targets.py
import argparse, pandas as pd, re

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
args = ap.parse_args()

df = pd.read_csv(args.file)

def norm_targets(s):
    if pd.isna(s): return ""
    toks = re.split(r"[,\s]+", str(s).strip())
    out = []
    for t in toks:
        t = t.strip().upper()
        if t in {"", "NONE"}: 
            continue
        if re.fullmatch(r"15\.[1-9]", t): 
            t = "SDG15." + t.split(".")[1]
        if re.fullmatch(r"SDG15\.(1|2|3|4|5|7|8|9)", t):
            out.append(t)
    return ",".join(sorted(set(out)))

df["sdg15_targets"] = df["sdg15_targets"].map(norm_targets)
df.to_csv(args.file, index=False)
print("normalized:", args.file)
