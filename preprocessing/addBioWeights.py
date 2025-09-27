from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd


# ---------- repo root ----------
ROOT = Path(__file__).resolve().parents[1]


# ---------- helpers ----------

def pick(df: pd.DataFrame, names: list[str]) -> str:
    """Pick the first column name that exists in df, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        c = lower_map.get(n.lower())
        if c is not None:
            return c
    raise KeyError(f"None of {names} found in columns: {list(df.columns)[:12]} ...")


def compute_bio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add biodiversity weight columns: binary (>=1) and weighted ({2:1.0,1:0.4,else 0})."""
    if "biodiversity" not in df.columns:
        raise KeyError("Required column 'biodiversity' not found.")

    biod = pd.to_numeric(df["biodiversity"], errors="coerce")
    bio_bin = (biod.fillna(0) >= 1).astype("UInt8")

    # Treat 2=principal, 1=significant; anything else -> 0.0
    wmap = {2: 1.0, 1: 0.4, 0: 0.0, 3: 0.0}
    bio_wtd = biod.map(wmap).fillna(0.0).astype("float32")

    out = df.copy()
    out["bio_weight_bin"] = bio_bin
    out["bio_weight_wtd"] = bio_wtd
    return out


def guess_dup_key(df: pd.DataFrame, year_c: str, focus_year: int = 2023) -> list[str] | None:
    """Pick a realistic dedup key. Prefer activity_id; otherwise try common CRS combos."""
    if "activity_id" in df.columns:
        return ["activity_id"]
    candidates = [
        ["crs_id"],
        ["CRS_ID"],
        ["CrsID"],
        ["project_number", "donor_code", "recipient_code", year_c],
        ["project_number", "donor_code", "recipient_code", "purpose_code", year_c],
        ["project_number", "donor_code", "recipient_code", "sector_code", year_c],
        ["project_number", "donor_code", "recipient_code", "agency_code", year_c],
        ["project_number", "donor_code", "recipient_code", "flow_code", year_c],
    ]
    present = set(df.columns)
    d = df[pd.to_numeric(df[year_c], errors="coerce") == focus_year].copy()
    for combo in candidates:
        if all(c in present for c in combo):
            # if this combo exists, use it; it's far better than falling back to ['year']
            return combo
    return None


def pick_money_column(df: pd.DataFrame) -> str | None:
    # Prefer deflated disbursements, then commitments.
    for c in ["usd_disbursement_defl", "usd_commitment_defl", "usd_disbursement", "usd_commitment"]:
        if c in df.columns:
            return c
    # Fallback: any usd_* column
    any_usd = [c for c in df.columns if c.lower().startswith("usd_")]
    return any_usd[0] if any_usd else None


def run_health_checks(df: pd.DataFrame, year_focus: int = 2023) -> None:
    year_c = pick(df, ["year"])
    bio_c  = pick(df, ["biodiversity", "rio_biodiversity_marker", "biodiversity_marker"])
    money_c = pick_money_column(df)

    tmp = df.copy()
    if "bio_weight_bin" not in tmp.columns or "bio_weight_wtd" not in tmp.columns:
        tmp = compute_bio_columns(tmp)

    # 0) Yearly totals
    y_int = pd.to_numeric(tmp[year_c], errors="coerce").astype("Int64")
    yr = (
        pd.DataFrame({
            "year": y_int,
            "bio_weight_bin": tmp["bio_weight_bin"].astype("float32"),
            "bio_weight_wtd": tmp["bio_weight_wtd"].astype("float32"),
        })
        .dropna(subset=["year"]).groupby("year", as_index=False)
        .sum(numeric_only=True).sort_values("year").set_index("year").round(1)
    )
    print("\n[BIO] Yearly totals (sum of bio_weight_*):")
    print(yr)

    # 1) Marker breakdown
    print("\n[Marker breakdown by year] counts of projects:")
    mk = (
        tmp.groupby([year_c, bio_c]).size()
        .unstack(fill_value=0)
        .rename(columns={2:"principal(2)", 1:"significant(1)", 0:"none(0)"}).sort_index()
    )
    print(mk)

    # 2) Dedup sanity with a proper key
    key = guess_dup_key(tmp, year_c, focus_year=year_focus)
    if key:
        dup_count = tmp[pd.to_numeric(tmp[year_c], errors="coerce") == year_focus].duplicated(subset=key).sum()
        print(f"\n[Duplicates in {year_focus}] duplicated rows on {key}: {dup_count}")
    else:
        print(f"\n[Duplicates in {year_focus}] skipped (no suitable key columns found).")

    # 3) Money vs counts
    if money_c:
        s_money = (
            tmp[tmp[bio_c] >= 1]
            .groupby(year_c)[money_c].sum(min_count=1)
            .rename("bio_money_sum")
        )
        counts = yr["bio_weight_bin"].rename("bio_proj_count")
        money_counts = pd.concat([s_money, counts], axis=1)
        print(f"\n[Money vs counts] using '{money_c}':")
        print(money_counts)
    else:
        usd_cols = [c for c in df.columns if c.lower().startswith("usd_")]
        print(f"\n[Money vs counts] No money column found; skipped. USD-like columns: {usd_cols}")

    # 4) Ratio
    ratio = (yr["bio_weight_wtd"] / yr["bio_weight_bin"]).rename("wtd_to_bin_ratio")
    print("\n[Weighted/Count ratio by year] (lower means larger share of 'significant(1)'):")
    print(ratio.round(3))


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Add biodiversity weight columns and write analysis-ready Parquet.")
    ap.add_argument(
        "--source",
        default=str(ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered_dedup.parquet"),
        help="Input Parquet (deduplicated dataset)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output Parquet path. If omitted, uses CRS_{ymin}_{ymax}_analysis_ready.parquet in the same directory as source.",
    )
    ap.add_argument(
        "--year-focus",
        type=int,
        default=2023,
        help="Year to use for duplicate sanity check (default: 2023).",
    )
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f"Not found: {src}")

    print(f"[BIO] Reading: {src.resolve()}")
    df = pd.read_parquet(src, engine="pyarrow")

    # Add biodiversity weight columns
    df = compute_bio_columns(df)

    # Determine year range for naming
    y = pd.to_numeric(df.get("year"), errors="coerce")
    ymin = int(y.min()) if y.notna().any() else None
    ymax = int(y.max()) if y.notna().any() else None

    # Resolve output path
    if args.output:
        out = Path(args.output)
    else:
        parent = src.parent
        out = parent / (f"CRS_{ymin}_{ymax}_analysis_ready.parquet" if ymin is not None and ymax is not None else "CRS_analysis_ready.parquet")

    # Save
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"[BIO] Saved -> {out.resolve()}")

    # Debug: list usd columns found in the saved file
    usd_cols = [c for c in df.columns if c.lower().startswith("usd_")]
    print(f"[BIO] USD columns present: {usd_cols}")

    # Run integrated health checks
    run_health_checks(df, year_focus=args.year_focus)


if __name__ == "__main__":
    main()
