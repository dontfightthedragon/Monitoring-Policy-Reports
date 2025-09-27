from pathlib import Path
import numpy as np, pandas as pd, random

def seed_all(s=1):
    import os
    os.environ["PYTHONHASHSEED"]=str(s)
    random.seed(s); np.random.seed(s)

def load_textprep(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    # prefer text_clean, fall back to text_en
    if "text_clean" in df: df["text_for_nlp"] = df["text_clean"]
    else: df["text_for_nlp"] = df["text_en"].astype(str)
    return df

def ensure_outdir(p: str) -> Path:
    pth = Path(p); pth.mkdir(parents=True, exist_ok=True); return pth
