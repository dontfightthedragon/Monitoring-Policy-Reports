#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
TABDIR = ROOT / "reports" / "tables"
QADIR  = TABDIR / "lda_qa"
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- Load LDA outputs ----
top_terms = pd.read_csv(TABDIR / "lda_top_terms.csv")                 # topic, top_terms, topic_label...
sizes     = pd.read_csv(QADIR / "topic_sizes.csv")                    # topic, count
money     = pd.read_csv(QADIR / "topic_disbursements.csv")            # topic, usd_disbursement_defl
prev_year = pd.read_csv(TABDIR / "lda_topic_prevalence_by_year.csv")  # year, topic columns

# Normalize money col name
if "usd_disb" not in money.columns and "usd_disbursement_defl" in money.columns:
    money = money.rename(columns={"usd_disbursement_defl": "usd_disb"})

# Build a topic -> label map (fallback to "Topic X" if missing)
label_map = {int(r.topic): (str(r.topic_label) if pd.notna(r.topic_label) and str(r.topic_label).strip()
                            else f"Topic {int(r.topic)}")
             for _, r in top_terms.iterrows()}

# ---- Table: top topics by disbursement (LaTeX) ----
money_sorted = money.sort_values("usd_disb", ascending=False)
top = money_sorted.merge(top_terms[["topic","topic_label","top_terms"]], on="topic", how="left")
top["usd_disb"] = top["usd_disb"].map(lambda x: f"{x:,.1f}")
top_out = top.head(10)[["topic","topic_label","usd_disb","top_terms"]]
(TABDIR / "lda_top_topics_by_money.tex").write_text(
    top_out.to_latex(index=False,
                     caption="Top LDA topics by total disbursements (USD, millions).",
                     label="tab:lda_top_money", escape=False)
)

# ---- Figure: disbursements by topic (bar) ----
money_plot = money_sorted.head(15).copy()
money_plot["label"] = money_plot["topic"].map(lambda t: f"{t}: {label_map.get(int(t), f'Topic {int(t)}')}")
plt.figure(figsize=(9, 6))
bars = plt.bar(money_plot["label"], money_plot["usd_disb"])
plt.xlabel("Topic")
plt.ylabel("Disbursements (USD, millions)")
plt.title("Top LDA topics by disbursements")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.3)
# annotate bars
for i, v in enumerate(money_plot["usd_disb"].tolist()):
    plt.text(i, v, f"{v:,.1f}", va="bottom", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIGDIR / "lda_topics_by_money.png", dpi=300)
plt.close()

# ---- Figure: topic prevalence over time (top 5 by money) ----
top5_topics = [int(t) for t in money_sorted.head(5)["topic"].tolist()]
sub = prev_year[["year"] + [str(t) for t in top5_topics]].copy()
# rename columns to "<id>: <label>"
rename_cols = {str(t): f"{t}: {label_map.get(t, f'Topic {t}')}" for t in top5_topics}
sub = sub.rename(columns=rename_cols)
sub["year"] = sub["year"].astype(int)

plt.figure(figsize=(9, 4.8))
for col in [c for c in sub.columns if c != "year"]:
    plt.plot(sub["year"], sub[col], marker="o", markersize=4, linewidth=2, label=col)
plt.legend(title=None)
plt.xlabel("Year")
plt.ylabel("Number of projects")
plt.title("Prevalence over time (Top 5 LDA topics by money)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "lda_prevalence_top5.png", dpi=300)
plt.close()

print("[LDA-REPORT] Wrote:")
print("  - tables/lda_top_topics_by_money.tex")
print("  - figures/lda_topics_by_money.png")
print("  - figures/lda_prevalence_top5.png")
