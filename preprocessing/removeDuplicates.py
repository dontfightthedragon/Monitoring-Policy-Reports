from __future__ import annotations
from pathlib import Path
import pandas as pd


# Paths (repo-root relative)
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered.parquet"
OUTPUT = ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered_dedup.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns: trim spaces, set dtypes, validate values, and
    optionally compact memory with categoricals.
    """
    df = df.copy()

    # Trim and collapse whitespace on text/categorical columns
    for c in ["donor_name", "recipient_name", "flow_name", "language"]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # Dtypes
    if "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        df["year"] = y.astype("Int64" if y.isna().any() else "int64")

    if "activity_id" in df.columns:
        df["activity_id"] = df["activity_id"].astype(str).str.strip()

    if "biodiversity" in df.columns:
        b = pd.to_numeric(df["biodiversity"], errors="coerce")
        b = b.where(b.isin([0, 1, 2, 3]))
        df["biodiversity"] = b.astype("Int64")

    # Optional: categories for memory footprint
    for c in ["donor_name", "recipient_name", "flow_name", "language"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DEDUP] Reading filtered dataset: {INPUT}")
    df = pd.read_parquet(INPUT, engine="pyarrow")

    # Ensure helper columns for sorting
    year_int = pd.to_numeric(df.get("year"), errors="coerce").fillna(-10**9).astype("int64")
    text_len = df.get("text", pd.Series(["" for _ in range(len(df))])).astype(str).str.len()

    # Prefer latest year, then longest text; stable sort to keep deterministic choice
    order_cols = pd.DataFrame({"_year": year_int, "_len": text_len})
    sort_index = order_cols.sort_values(["_year", "_len"], ascending=[False, False]).index
    df_sorted = df.loc[sort_index]

    # Drop duplicates by activity_id keeping the best-ranked row
    if "activity_id" not in df_sorted.columns:
        raise KeyError("'activity_id' column missing in filtered dataset; cannot deduplicate.")

    before = len(df_sorted)
    dedup = df_sorted.drop_duplicates(subset=["activity_id"], keep="first").copy()
    removed = before - len(dedup)
    print(f"[DEDUP] Removed {removed:,} duplicate rows (by activity_id)")

    # Normalize columns/dtypes for downstream usage
    dedup = _normalize(dedup)

    # Save
    dedup.to_parquet(OUTPUT, engine="pyarrow", index=False)
    print(f"[DEDUP] Saved -> {OUTPUT}")
    print(f"[DEDUP] Rows (post-dedup): {len(dedup):,}")


if __name__ == "__main__":
    main()
