# Creates a clean pipeline overview figure (PDF only).
# Saves to: analysis/LLM/figures/pipeline_overview.pdf

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from matplotlib.lines import Line2D

out_dir = "analysis/LLM/figures"
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "pipeline_overview.pdf")

def box(ax, xy, text, width=2.8, height=0.9, fontsize=11, lw=1.2, fc="#F5F5F5"):
    x, y = xy
    p = FancyBboxPatch((x, y), width, height,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       edgecolor="black", facecolor=fc, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)
    return p

def arrow(ax, p_from, p_to, lw=1.2):
    x1 = p_from.get_x() + p_from.get_width()
    y1 = p_from.get_y() + p_from.get_height()/2
    x2 = p_to.get_x()
    y2 = p_to.get_y() + p_to.get_height()/2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=ArrowStyle("Simple", head_width=6, head_length=8),
                                lw=lw, color="black", shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=0.0"))

plt.rcParams["figure.dpi"] = 300
fig, ax = plt.subplots(figsize=(15, 3.6))
ax.axis("off")

x0, step, y = 0.3, 2.5, 1.2

p_crs   = box(ax, (x0 + 0*step, y), "CRS data\n(ODA, 2015–2023)")
p_pre   = box(ax, (x0 + 1*step, y), "Preprocessing\nfilter • dedup • translate")
p_text  = box(ax, (x0 + 2*step, y), "Consolidated text\n(text_for_llm)")
p_pref  = box(ax, (x0 + 3*step, y), "Anchor prefilter\n(high-precision terms)")
p_llm   = box(ax, (x0 + 4*step, y), "LLM classification\nGPT-4 • GPT-3.5")
p_json  = box(ax, (x0 + 5*step, y), "Strict JSON\nprimary • targets • rationale")
p_post  = box(ax, (x0 + 6*step, y), "Postprocessor\nadd-only trigger augmentation")
p_eval  = box(ax, (x0 + 7*step, y), "Aggregation & evaluation\nagreement • κ • JSD • Jaccard")

for a, b in [(p_crs,p_pre),(p_pre,p_text),(p_text,p_pref),(p_pref,p_llm),
             (p_llm,p_json),(p_json,p_post),(p_post,p_eval)]:
    arrow(ax, a, b)

ax.add_line(Line2D([0.02, 0.02], [0.06, 0.06], lw=1.2, color="black"))
ax.text(0.04, 0.06,
        "Deterministic (temperature=0); caching at each stage; reproducible artifacts.",
        transform=ax.transAxes, fontsize=9, va="center")

ax.text(0.5, 0.94, "Overview of the Biodiversity Project Classification Pipeline",
        transform=ax.transAxes, ha="center", va="center", fontsize=13)

plt.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {pdf_path}")
