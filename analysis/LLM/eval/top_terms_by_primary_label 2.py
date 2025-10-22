# analysis/LLM/eval/top_terms_by_primary_label.py
import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

INP = "analysis/LLM/artifacts/full/llm_predictions_w_text.csv"
df = pd.read_csv(INP).fillna("")

# simple cleaner
def norm(s): return re.sub(r"[^a-z0-9\s]", " ", s.lower())

labels = ["FORESTS_LAND","WATER_MARINE","WILDLIFE_SPECIES","CROSS_CUTTING","NONE"]
X_text = df["text_for_llm"].map(norm)
y = df["llm_primary_label"].where(df["llm_primary_label"].isin(labels), "NONE")

# unigrams+bigrams; drop very rare terms
cv = CountVectorizer(ngram_range=(1,2), min_df=50, stop_words="english")
X = cv.fit_transform(X_text)
terms = np.array(cv.get_feature_names_out())

for lab in labels:
    y_bin = (y == lab).astype(int)
    chi, _ = chi2(X, y_bin)
    top_idx = chi.argsort()[::-1][:50]
    pd.DataFrame({"term": terms[top_idx], "chi2": chi[top_idx]}) \
      .to_csv(f"analysis/LLM/eval/top_terms_{lab}.csv", index=False)

print("wrote analysis/LLM/eval/top_terms_*.csv")
