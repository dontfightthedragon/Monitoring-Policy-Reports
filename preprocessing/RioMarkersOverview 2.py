# preprocessing/RioMarkersOverview.py
"""
Compute distributions and presence for Rio marker columns.

Usage:
  python preprocessing/RioMarkersOverview.py \
    --input dataset/rawDataset/CRS.parquet \
    --outdir reports/tables \
    --markers biodiversity climate_mitigation climate_adaptation desertification
"""
import argparse
from pathlib import Path
import pandas as pd


def resolve_path(p: str) -> Path:
    pth = Path(p).expanduser().resolve()
    if not pth.exists():
        raise FileNotFoundError(f"Input file not found: {pth}")
    return pth


def ensure_outdir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rio marker distribution & presence summary")
    ap.add_argument(
        "--input", required=True, help="Input CRS parquet file (pyarrow-readable)"
    )
    ap.add_argument(
        "--outdir", default="reports/tables", help="Directory to write CSV outputs"
    )
    ap.add_argument(
        "--markers",
        nargs="*",
        default=["biodiversity", "climate_mitigation", "climate_adaptation", "desertification"],
        help="Marker columns to analyze (only those present in the dataset will be used)",
    )
    ap.add_argument(
        "--engine",
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Parquet engine",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inpath = resolve_path(args.input)
    out_dir = ensure_outdir(args.outdir)

    print(f"Input: {inpath}")
    print(f"Output directory: {out_dir}")

    df = pd.read_parquet(inpath, engine=args.engine)
    print(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")

    # Filter to markers that actually exist
    marker_cols = [c for c in args.markers if c in df.columns]
    missing_cols = [c for c in args.markers if c not in df.columns]
    if missing_cols:
        print(f"Skipping missing columns: {missing_cols}")
    if not marker_cols:
        raise SystemExit("No specified marker columns found in dataset.")

    # Distributions and presence
    all_vc = []
    presence_rows = []
    for col in marker_cols:
        vc = df[col].value_counts(dropna=False).sort_index()
        all_vc.append(
            vc.rename_axis("value").reset_index(name="count").assign(marker=col)
        )
        total = len(df)
        present = df[col].notna().sum()
        presence_rows.append(
            {
                "marker": col,
                "present_n": int(present),
                "present_pct": round(100 * present / total, 2),
                "missing_n": int(total - present),
                "missing_pct": round(100 * (total - present) / total, 2),
            }
        )
        print(f"\n{col} value counts:")
        print(vc)

    vc_df = pd.concat(all_vc, ignore_index=True)[["marker", "value", "count"]]
    presence_df = pd.DataFrame(presence_rows)

    # Optional human-readable labels for common Rio codes
    code_map = {0: "not targeted", 1: "significant", 2: "principal"}
    vc_labeled = vc_df.copy()
    vc_labeled["value_label"] = vc_labeled["value"].map(code_map).fillna(
        vc_labeled["value"].astype(str)
    )

    # Save
    vc_df.to_csv(out_dir / "rio_marker_value_counts.csv", index=False)
    vc_labeled.to_csv(out_dir / "rio_marker_value_counts_labeled.csv", index=False)
    presence_df.to_csv(out_dir / "rio_marker_presence.csv", index=False)

    print("\nSaved:")
    print(f" - {out_dir / 'rio_marker_value_counts.csv'}")
    print(f" - {out_dir / 'rio_marker_value_counts_labeled.csv'}")
    print(f" - {out_dir / 'rio_marker_presence.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
