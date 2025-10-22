# analysis/LLM/plots/targets_by_year_top5.py
import pandas as pd, matplotlib.pyplot as plt

tg = pd.read_csv("analysis/LLM/artifacts/full/summary_targets_by_year.csv")

# pick top 5 by total count
top5 = (tg.groupby("target")["n"].sum().sort_values(ascending=False).head(5).index.tolist())
t5 = tg[tg["target"].isin(top5)]
p = t5.pivot(index="year", columns="target", values="share").fillna(0)

plt.figure()
p.sort_index().plot(ax=plt.gca())
plt.ylabel("Share of SDG15 target assignments")
plt.xlabel("Year")
plt.title("SDG15 targets over time (top 5)")
plt.legend(title="Target")
plt.tight_layout()
plt.savefig("analysis/LLM/plots/targets_by_year_top5.pdf")
