
from pathlib import Path
import pandas as pd

def read_crs_parquet(path, columns=None):
    path = Path(path)
    return pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)

def save_parquet(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
