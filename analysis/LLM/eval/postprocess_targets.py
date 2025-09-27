# analysis/LLM/eval/postprocess_targets.py
import sys, re, pandas as pd

INP, OUTP = sys.argv[1], sys.argv[2]
df = pd.read_csv(INP)  # needs: activity_id, llm_primary_label, llm_sdg15_targets, llm_rationale, text_for_llm

# regex cues (precise phrases only)
R = {
    "SDG15.1": re.compile(r"\b(wetland|ramsar|river|lake|freshwater|floodplain|riparian|catchment)\b", re.I),
    "SDG15.2": re.compile(r"\b(reforest|afforest|sustainable forest management|forest management|illegal logging|redd\+)\b", re.I),
    "SDG15.3": re.compile(r"\b(land degradation|desertification|restore degraded land|erosion control|rangeland restoration|soil restoration)\b", re.I),
    "SDG15.4": re.compile(r"\b(mountain|alpine)\b", re.I),
    "SDG15.5": re.compile(r"\b(biodiversity loss|species (decline|recovery)|habitat (loss|restoration)|key biodiversity area|kba)\b", re.I),
    "SDG15.7": re.compile(r"\b(poaching|anti-?poaching|wildlife trafficking|cites enforcement?)\b", re.I),
    "SDG15.8": re.compile(r"\b(invasive (alien )?species|biosecurity)\b", re.I),
    "SDG15.9": re.compile(r"\b(nbsap|mainstreaming biodiversity|biodiversity (policy|planning)|spatial planning|land-use planning)\b", re.I),
}

def merge_targets(pred: str, text: str) -> str:
    ts = {t.strip() for t in str(pred or "").split(",") if t.strip()}
    t = text or ""
    for code, rx in R.items():
        if rx.search(t):
            ts.add(code)
    # keep only valid SDG15.x
    ts = sorted({x for x in ts if re.fullmatch(r"SDG15\.(1|2|3|4|5|7|8|9)", x)})
    return ",".join(ts)

df["llm_sdg15_targets"] = df.apply(
    lambda r: merge_targets(r.get("llm_sdg15_targets",""), r.get("text_for_llm","")),
    axis=1
)

# write back in the same schema as predictions (drop text column)
cols = ["activity_id","llm_primary_label","llm_sdg15_targets","llm_rationale"]
df[cols].to_csv(OUTP, index=False)
print("wrote", OUTP)
