from pathlib import Path
import pandas as pd

# ====== EDIT THESE 3 LINES ======
# Read the current filtered dataset written by the preprocessing step.
input_path = Path("dataset/temporaryFileToWorkWith/CRS_filtered.parquet")
output_dir = Path("reports/tables")
parquet_engine = None   # set to "pyarrow" or "fastparquet" if you prefer; None = auto



def human_bytes(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading ===")
    print(f"File: {input_path.resolve()}")
    try:
        df = pd.read_parquet(input_path, engine=parquet_engine) if parquet_engine else pd.read_parquet(input_path)
    except Exception as e:
        print(f"Failed to read Parquet: {e}")
        return

    n_rows, n_cols = df.shape
    mem_bytes = df.memory_usage(deep=True).sum()
    print(f"Rows: {n_rows:,} | Cols: {n_cols} | Est. memory: {human_bytes(mem_bytes)}")
    print("Columns:", list(df.columns))

    # === Year coverage ===
    print("\n=== Year coverage ===")
    if "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype("int64")
        if not years.empty:
            print(f"Years range: {years.min()}–{years.max()} | Unique years: {years.nunique()}")
            per_year = years.value_counts().sort_index()
            print(per_year.tail(15))
            per_year.rename_axis("year").reset_index(name="n").to_csv(output_dir / "per_year_counts.csv", index=False)
        else:
            print("Could not parse 'year' to integers.")
    else:
        print("Column 'year' not found.")

    # === Language distribution ===
    print("\n=== Language distribution ===")
    if "language" in df.columns:
        lang_counts = df["language"].value_counts(dropna=False)
        print(lang_counts.head(20))
        lang_counts.rename_axis("language").reset_index(name="n").to_csv(output_dir / "language_counts.csv", index=False)
    else:
        print("Column 'language' not found.")

    # === Biodiversity distribution ===
    print("\n=== Biodiversity distribution ===")
    if "biodiversity" in df.columns:
        biodist = df["biodiversity"].value_counts(dropna=False).sort_index()
        print(biodist)
        biodist.rename_axis("biodiversity").reset_index(name="n").to_csv(output_dir / "biodiversity_counts.csv", index=False)
    else:
        print("Column 'biodiversity' not found.")

    # === Top donors / recipients ===
    print("\n=== Top donors / recipients ===")
    if "donor_name" in df.columns:
        donors_top = df["donor_name"].value_counts().head(20)
        print("\nTop 20 donors:")
        print(donors_top)
        donors_top.rename_axis("donor_name").reset_index(name="n").to_csv(output_dir / "top20_donors.csv", index=False)
    else:
        print("Column 'donor_name' not found.")

    if "recipient_name" in df.columns:
        recipients_top = df["recipient_name"].value_counts().head(20)
        print("\nTop 20 recipients:")
        print(recipients_top)
        recipients_top.rename_axis("recipient_name").reset_index(name="n").to_csv(output_dir / "top20_recipients.csv", index=False)
    else:
        print("Column 'recipient_name' not found.")

    # === Recent 5 years (optional quick lens) ===
    if "year" in df.columns and "donor_name" in df.columns:
        years_int = pd.to_numeric(df["year"], errors="coerce").dropna().astype("int64")
        if not years_int.empty:
            last5_min = years_int.max() - 4
            recent = df[years_int.index.isin(df.index) & (years_int >= last5_min)]
            if not recent.empty:
                recent["donor_name"].value_counts().head(20).rename_axis("donor_name").reset_index(name="n") \
                    .to_csv(output_dir / "top20_donors_recent5y.csv", index=False)

    # === Text length stats ===
    print("\n=== Text length (chars) ===")
    if "text" in df.columns:
        lengths = df["text"].astype(str).str.len()
        q = lengths.quantile([0.0, 0.25, 0.5, 0.9, 0.99, 1.0])
        print(q.to_string())
        pd.DataFrame({"quantile": q.index, "char_len": q.values}).to_csv(output_dir / "text_length_quantiles.csv", index=False)
        print(f"Average length: {lengths.mean():.1f} | Std: {lengths.std():.1f}")
    else:
        print("Column 'text' not found.")

    # === Null rate per column ===
    print("\n=== Null rate by column (%) ===")
    null_rate = (df.isna().sum() * 100.0 / len(df)).sort_values(ascending=False)
    print(null_rate.head(20))
    null_rate.rename_axis("column").reset_index(name="null_pct").to_csv(output_dir / "null_rate.csv", index=False)

    # === Duplicate activity_id check ===
    print("\n=== Duplicate activity_id check ===")
    if "activity_id" in df.columns:
        dup_rows_mask = df.duplicated(subset=["activity_id"], keep=False)
        n_dup_rows = int(dup_rows_mask.sum())
        n_dup_ids = df.loc[dup_rows_mask, "activity_id"].nunique()
        print(f"Duplicate rows by activity_id: {n_dup_rows:,} | Distinct duplicated IDs: {n_dup_ids:,}")
        if n_dup_rows > 0:
            (df.loc[dup_rows_mask, ["activity_id", "year", "donor_name", "recipient_name"]]
               .head(1000)
               .to_csv(output_dir / "duplicate_activity_id_sample.csv", index=False))
    else:
        print("Column 'activity_id' not found.")

    # === Sample rows ===
    print("\n=== Sample rows (head) ===")
    sample_cols = [c for c in ["year","activity_id","donor_name","recipient_name","language","biodiversity","text"] if c in df.columns]
    print(df[sample_cols].head(5).to_string(index=False))

    print(f"\nSaved table summaries to: {output_dir.resolve()}")
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
