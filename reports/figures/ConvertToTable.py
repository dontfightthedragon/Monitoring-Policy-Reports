import pandas as pd

# Load CSV
df = pd.read_csv("top_recipients_analysis.csv")

# Convert to LaTeX
print(df.to_latex(index=False, caption="Top recipients of biodiversity funding, 2015–2023", label="tab:top_recipients"))
