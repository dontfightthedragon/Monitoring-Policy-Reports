from pathlib import Path
import hashlib
import json
import os
import subprocess
import pandas as pd
from datetime import datetime

# EDIT if paths differ
SOURCE = Path("dataset/temporaryFileToWorkWith/CRS_filtered_dedup.parquet")
CANON_DIR = Path("dataset/canonical")

# Version tag = today; change if you want semantic versions.
VERSION = datetime.now().strftime("v%Y-%m-%d")
TARGET = CANON_DIR / f"CRS_canonical_{VERSION}.parquet"
MANIFEST = CANON_DIR / f"CRS_canonical_{VERSION}.manifest.json"
LATEST = CANON_DIR / "latest.parquet"

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Source does not exist: {SOURCE}")

    CANON_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Copy file (avoid re-encoding)
    if TARGET.exists():
        raise SystemExit(f"Target already exists (won't overwrite): {TARGET}")
    data = pd.read_parquet(SOURCE)  # also a good moment to sanity check
    data.to_parquet(TARGET, index=False)  # raw copy via pandas; DuckDB COPY also fine

    # 2) Compute checksum & basic stats
    checksum = sha256_file(TARGET)
    n_rows, n_cols = data.shape
    dtypes = {c: str(t) for c, t in data.dtypes.items()}

    # Simple sanity metrics
    years = pd.to_numeric(data["year"], errors="coerce") if "year" in data.columns else None
    years_min = int(years.min()) if years is not None and years.notna().any() else None
    years_max = int(years.max()) if years is not None and years.notna().any() else None
    dup_activity = int(data.duplicated(subset=["activity_id"]).sum()) if "activity_id" in data.columns else None
    # Ensure JSON-serializable keys (avoid numpy.int64/NaN as dict keys)
    if "biodiversity" in data.columns:
        vc = data["biodiversity"].value_counts(dropna=False)
        def _json_key(k):
            # Represent missing as string; JSON object keys must be strings
            if pd.isna(k):
                return "null"
            # Prefer native Python int when possible; fallback to string
            try:
                return int(k)
            except Exception:
                return str(k)
        biod_counts = { _json_key(k): int(v) for k, v in vc.items() }
    else:
        biod_counts = None

    manifest = {
        "name": TARGET.name,
        "path": str(TARGET),
        "version": VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "sha256": checksum,
        "rows": n_rows,
        "cols": n_cols,
        "dtypes": dtypes,
        "years_min": years_min,
        "years_max": years_max,
        "duplicate_activity_id_rows": dup_activity,
        "biodiversity_value_counts": biod_counts,
        "source_path": str(SOURCE.resolve()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # 3) Point stable handle to this version
    try:
        if LATEST.exists() or LATEST.is_symlink():
            LATEST.unlink()
        os.symlink(TARGET.name, LATEST)  # relative symlink inside same folder
    except Exception:
        # On Windows or restricted FS, fall back to making a copy
        import shutil
        shutil.copy2(TARGET, LATEST)

    print(f"Frozen canonical file:\n  {TARGET}")
    print(f"Manifest:\n  {MANIFEST}")
    print(f"'latest.parquet' -> {TARGET.name}")
    print(f"SHA256: {checksum}")
    print(f"Rows x Cols: {n_rows} x {n_cols}")
    print(f"Years: {years_min}–{years_max}")
    print(f"Duplicate activity_id rows: {dup_activity}")
    print("Done.")
    
if __name__ == "__main__":
    main()
