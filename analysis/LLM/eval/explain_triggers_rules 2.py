# analysis/LLM/eval/explain_triggers_rules.py
import re, pandas as pd
from pathlib import Path

INP = Path("analysis/LLM/artifacts/full/llm_predictions_w_text.csv")  # must include text_for_llm
df = pd.read_csv(INP, dtype=str).fillna("")

# same cues as in postprocess_targets.py (edit if you changed that file)
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

rows=[]
for _, r in df.iterrows():
    txt = r.get("text_for_llm","")
    for code, rx in R.items():
        for m in rx.finditer(txt):
            phrase = m.group(0).lower()
            rows.append({
                "activity_id": r["activity_id"],
                "llm_primary_label": r.get("llm_primary_label",""),
                "llm_sdg15_targets": r.get("llm_sdg15_targets",""),
                "target_trigger": code,
                "matched_phrase": phrase
            })

out = pd.DataFrame(rows)
out.to_csv("analysis/LLM/eval/trigger_hits_per_row.csv", index=False)

agg = (out.groupby(["target_trigger","matched_phrase"])
          .size().rename("count").reset_index()
          .sort_values(["target_trigger","count"], ascending=[True, False]))
agg.to_csv("analysis/LLM/eval/trigger_hits_summary.csv", index=False)
print("wrote trigger_hits_per_row.csv and trigger_hits_summary.csv")

