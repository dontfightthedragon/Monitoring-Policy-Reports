# analysis/LLM/eval/split_sets.py
import pandas as pd, numpy as np
from pathlib import Path

RNG = 7
N_TEST = 50
GOLD = Path("analysis/LLM/artifacts/gold/gold_150.csv")
OUTD = Path("analysis/LLM/artifacts/gold"); OUTD.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(GOLD)
counts = df["primary_label"].value_counts()
props  = counts / counts.sum()
q = (props * N_TEST).apply(np.floor).astype(int)
# distribute leftover by largest fractional parts
left = N_TEST - int(q.sum())
frac = (props * N_TEST - q).sort_values(ascending=False)
q.loc[frac.index[:left]] += 1

np.random.seed(RNG)
test_idx = []
for lab, n in q.items():
    idx = df.index[df["primary_label"] == lab]
    take = min(int(n), len(idx))
    if take > 0:
        test_idx.extend(np.random.choice(idx, size=take, replace=False))

test = df.loc[sorted(set(test_idx))]
dev  = df.drop(test.index)

test.to_csv(OUTD/"test.csv", index=False)
dev.to_csv(OUTD/"dev.csv", index=False)
print(len(test), len(dev))
