# analysis/LLM/plots/donor_bio_rate.py
import re
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



# --- load
dprim = pd.read_csv("analysis/LLM/artifacts/full/summary_primary_by_donor.csv")

# --- keep bilateral donors only (drop multilaterals/funds/banks/etc.)
MULTILATERAL_EXACT = {
    "EU Institutions","European Commission","European Union",
    "International Development Association","World Bank",
    "Asian Development Bank","African Development Bank","Inter-American Development Bank",
    "Islamic Development Bank","Global Environment Facility",
    "UNDP","UNEP","UNICEF","UNESCO","FAO","IFAD","WFP","WHO","UNFPA","UNAIDS","UN Women",
}
MULTILATERAL_RX = re.compile(
    r"(Bank|Development Bank|Association|Facility|Fund|Commission|Council|Union|"
    r"Secretariat|Organization|Organisation|Agency|Authority|Corporation|Foundation|"
    r"United Nations|World Bank|GEF|IDA)", re.I
)

def is_multilateral(name: str) -> bool:
    if pd.isna(name): return True
    n = str(name).strip()
    return (n in MULTILATERAL_EXACT) or bool(MULTILATERAL_RX.search(n))

bilat = dprim[~dprim["donor"].map(is_multilateral)].copy()

# --- compute rate (share non-NONE)
tot = bilat.groupby("donor")["n"].sum().rename("tot")
bio = bilat[bilat["llm_primary_label"]!="NONE"].groupby("donor")["n"].sum().rename("bio")
m = pd.concat([tot, bio], axis=1).fillna(0.0)
m["rate"] = m["bio"] / m["tot"]

# top 15 bilateral donors by total volume, then sort by rate for the bar order
top = (m.sort_values("tot", ascending=False).head(15)
         .sort_values("rate", ascending=True))

# --- plot (no clipping)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top.index, top["rate"].values)
ax.set_xlim(0, 1)                      
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Share of projects classified as biodiversity-relevant")
ax.set_title("Biodiversity relevance rate by donor (bilateral; top 15 by volume)")
ax.set_xlim(0, 1)
# more room for long donor names
fig.subplots_adjust(left=0.35, right=0.98, top=0.90, bottom=0.10)

# save (bbox_inches='tight' avoids text cutoff)
out_pdf = "analysis/LLM/plots/donor_bio_rate_bilateral.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
