# preprocessing/RawDatasetOverview.py
import pandas as pd
from pathlib import Path

def resolve_path(p: str) -> Path:
    pth = Path(p).expanduser().resolve()
    if not pth.exists():
        raise FileNotFoundError(f"Input file not found: {pth}")
    return pth

def ensure_outdir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def main():
    # === CONFIG ===
    input_path = "/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/dataset/rawDataset/CRS.parquet"
    output_dir = "reports/tables"
    text_columns = ["short_description", "long_description", "project_title"]
    marker_columns = ["biodiversity"]
    sample_size = 5
    random_state = 1
    # ==============

    inpath = resolve_path(input_path)
    out_dir = ensure_outdir(output_dir)

    print(f"Input: {inpath}")
    print(f"Output directory: {out_dir}")

    # Load full schema (overview) – uses pyarrow
    df = pd.read_parquet(inpath, engine="pyarrow")
    print(f"Successfully loaded dataset: {len(df):,} rows, {len(df.columns)} columns")

    # Basic overview
    print("\n=== Dataset Overview ===")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"Approximate memory usage: {mem_gb:.2f} GB")

    # Text columns
    print("\n=== Text Columns ===")
    for col in text_columns:
        if col in df.columns:
            non_empty = df[col].notna().sum()
            pct = (non_empty / len(df) * 100.0) if len(df) else 0.0
            print(f"{col} non-empty: {non_empty:,} ({pct:.1f}%)")

            if non_empty:
                samples = df[col].dropna().sample(n=min(sample_size, non_empty), random_state=random_state)
                print(f"\n--- Sample {col} ---")
                for idx, text in enumerate(samples, 1):
                    s = str(text)
                    print(f"{idx}. {s[:200]}{'...' if len(s) > 200 else ''}")
                samples.to_csv(out_dir / f"sample_{col.lower()}.csv", index=False)
        else:
            print(f"{col} not found in dataset.")

    # Environmental markers
    print("\n=== Environmental Markers ===")
    for col in marker_columns:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).sort_index()
            print(f"\n{col} value counts:")
            print(vc)
            vc.rename_axis(col).reset_index(name="count").to_csv(out_dir / f"marker_{col}.csv", index=False)
        else:
            print(f"{col} not found in dataset.")

    # Year distribution
    if "year" in df.columns:
        print("\n=== Year Distribution ===")
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
        if not years.empty:
            year_counts = years.value_counts().sort_index()
            print(f"Year range: {years.min()}–{years.max()} | Unique years: {years.nunique()}")
            print(year_counts.tail(10))
            year_counts.rename_axis("year").reset_index(name="count").to_csv(out_dir / "docs_per_year.csv", index=False)
        else:
            print("No valid year data found.")
    else:
        print("\nYear column not found.")

    print(f"\n=== Done ===\nOutput files saved to: {out_dir}")

if __name__ == "__main__":
    main()
