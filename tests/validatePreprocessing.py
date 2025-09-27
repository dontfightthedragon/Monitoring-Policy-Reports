# tests/validate_crs_pipeline.py
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
from typing import List, Optional
import duckdb
import pandas as pd

def resolve(p: str) -> Path:
    q = Path(os.path.expanduser(os.path.expandvars(p)))
    if not q.is_absolute():
        q = (Path.cwd() / q).resolve(strict=False)
    if not q.exists():
        raise FileNotFoundError(f"Not found: {p} -> {q}")
    return q

def to_bool(v: Optional[str]) -> Optional[bool]:
    if v is None: return None
    return str(v).lower() in {"1","true","yes","y"}

def parse_list(v: Optional[str]) -> Optional[List[str]]:
    if not v: return None
    return [x.strip() for x in v.split(",") if x.strip()]

def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    ap = argparse.ArgumentParser(description="Validate CRS preprocessing outputs.")
    ap.add_argument("--filtered", required=True, help="Path to CRS_filtered.parquet")
    ap.add_argument("--dedup", required=True, help="Path to CRS_filtered_dedup.parquet")
    ap.add_argument("--report-dir", default="reports/qa", help="Where to write QA outputs")
    # Expectations (optional but recommended)
    ap.add_argument("--expect-oda", type=str, default=None, help="true/false: expect flow_name to contain 'ODA' in filtered file")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--allow-lang", type=str, default="en,fr", help="comma list; set empty to allow any")
    ap.add_argument("--min-text", type=int, default=20, help="min chars expected in text")
    ap.add_argument("--allow-biod", type=str, default="0,1,2,3", help="allowed biodiversity values (comma list) plus NULL")
    ap.add_argument("--sample-n", type=int, default=200, help="rows to save per failure case")
    ap.add_argument("--fail-on-error", action="store_true", help="exit(1) if any check fails")
    args = ap.parse_args()

    filtered = resolve(args.filtered)
    dedup = resolve(args.dedup)
    outdir = ensure_outdir(Path(args.report_dir))
    allow_lang = set([x for x in parse_list(args.allow_lang) or []])
    allow_biod = set([int(x) for x in parse_list(args.allow_biod) or []])
    expect_oda = to_bool(args.expect_oda)

    con = duckdb.connect()
    metrics = {}
    failures = []

    def record_fail(name:str, detail:str, df: Optional[pd.DataFrame] = None):
        print(f"FAIL: {name} -> {detail}")
        failures.append({"name": name, "detail": detail})
        if df is not None and len(df):
            df.head(args.sample_n).to_csv(outdir / f"{name}.csv", index=False)

    # ---------- Basic presence ----------
    for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
        try:
            schema = con.sql(f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0").df()
            cols = set(schema.columns)
            req = {"year","activity_id","donor_name","recipient_name","text","biodiversity","language"}
            missing = sorted(list(req - cols))
            if missing:
                record_fail(f"{label}_missing_columns", f"missing {missing}")
            metrics[f"{label}_columns"] = sorted(list(cols))
        except Exception as ex:
            record_fail(f"{label}_read_error", str(ex))
            print("Aborting due to read error.")
            con.close()
            if args.fail_on_error: sys.exit(1)
            else: sys.exit(0)

    # ---------- Row counts ----------
    for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
        n = int(con.sql(f"SELECT COUNT(*) AS n FROM read_parquet('{path.as_posix()}')").df()["n"][0])
        metrics[f"{label}_rows"] = n
        print(f"{label}: {n:,} rows")

    # ---------- Year bounds ----------
    for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
        ystats = con.sql(f"""
            SELECT MIN(CAST(year AS INT)) AS y_min,
                   MAX(CAST(year AS INT)) AS y_max,
                   COUNT(*) AS n
            FROM read_parquet('{path.as_posix()}')
            WHERE try_cast(year AS INT) IS NOT NULL
        """).df().iloc[0]
        metrics[f"{label}_year_min"] = int(ystats.y_min) if ystats.y_min is not None else None
        metrics[f"{label}_year_max"] = int(ystats.y_max) if ystats.y_max is not None else None
        if args.year_from is not None and (ystats.y_min is None or ystats.y_min < args.year_from):
            record_fail(f"{label}_year_from", f"min year {ystats.y_min} < expected {args.year_from}")
        if args.year_to is not None and (ystats.y_max is None or ystats.y_max > args.year_to):
            record_fail(f"{label}_year_to", f"max year {ystats.y_max} > expected {args.year_to}")

    # ---------- ODA-only expectation on filtered ----------
    if expect_oda is not None:
        if "flow_name" in metrics["filtered_columns"]:
            bad = con.sql(f"""
                SELECT flow_name, COUNT(*) AS n
                FROM read_parquet('{filtered.as_posix()}')
                WHERE flow_name IS NOT NULL
                  AND lower(flow_name) NOT LIKE '%%oda%%'
                GROUP BY 1 ORDER BY 2 DESC
            """).df()
            if expect_oda and len(bad):
                record_fail("filtered_not_oda", f"{int(bad['n'].sum())} rows where flow_name does not contain 'ODA'", bad)
        else:
            print("Note: 'flow_name' not present in filtered file; skipping ODA check.")

    # ---------- Min text length ----------
    for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
        short = con.sql(f"""
            SELECT activity_id, length(text) AS txtlen
            FROM read_parquet('{path.as_posix()}')
            WHERE text IS NULL OR length(text) < {int(args.min_text)}
        """).df()
        metrics[f"{label}_short_text"] = int(len(short))
        if len(short):
            record_fail(f"{label}_short_text", f"{len(short)} rows have text length < {args.min_text}", short)

    # ---------- Language allow-list ----------
    if allow_lang:
        for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
            bad = con.sql(f"""
                SELECT language, COUNT(*) AS n
                FROM read_parquet('{path.as_posix()}')
                WHERE language IS NOT NULL AND language NOT IN ({','.join([f"'{x}'" for x in allow_lang])})
                GROUP BY 1 ORDER BY 2 DESC
            """).df()
            if len(bad):
                record_fail(f"{label}_bad_language",
                            f"{int(bad['n'].sum())} rows not in allowed {sorted(list(allow_lang))}", bad)

    # ---------- Biodiversity values + null rate ----------
    for path, label in [(filtered,"filtered"), (dedup,"dedup")]:
        nulls = int(con.sql(f"SELECT COUNT(*) AS n FROM read_parquet('{path.as_posix()}') WHERE biodiversity IS NULL").df()["n"][0])
        total = metrics[f"{label}_rows"]
        metrics[f"{label}_biod_nulls"] = nulls
        metrics[f"{label}_biod_null_rate"] = nulls / total if total else None
        badvals = con.sql(f"""
            SELECT biodiversity AS biod, COUNT(*) AS n
            FROM read_parquet('{path.as_posix()}')
            WHERE biodiversity IS NOT NULL
              AND biodiversity NOT IN ({','.join(map(str, allow_biod))})
            GROUP BY 1 ORDER BY 2 DESC
        """).df()
        if len(badvals):
            record_fail(f"{label}_biod_bad_values",
                        f"found values outside allowed {sorted(list(allow_biod))}", badvals)

    # ---------- Duplicates by activity_id ----------
    # filtered: duplicates are allowed (pre-dedup), but we still report them
    dup_filtered = con.sql(f"""
        SELECT activity_id, COUNT(*) AS c
        FROM read_parquet('{filtered.as_posix()}')
        GROUP BY 1 HAVING COUNT(*) > 1
        ORDER BY c DESC LIMIT {args.sample_n}
    """).df()
    metrics["filtered_duplicate_groups"] = int(len(dup_filtered))
    if len(dup_filtered):
        dup_cnt = con.sql(f"""
            SELECT SUM(c) AS total_dup_rows FROM (
                SELECT COUNT(*) AS c
                FROM read_parquet('{filtered.as_posix()}')
                GROUP BY activity_id HAVING COUNT(*) > 1
            )
        """).df().iloc[0,0]
        record_fail("filtered_has_duplicates", f"{int(dup_cnt)} duplicate rows across {len(dup_filtered)} groups", dup_filtered)

    # dedup: duplicates should be zero
    dup_dedup = con.sql(f"""
        SELECT activity_id, COUNT(*) AS c
        FROM read_parquet('{dedup.as_posix()}')
        GROUP BY 1 HAVING COUNT(*) > 1
        LIMIT 1
    """).df()
    if len(dup_dedup):
        record_fail("dedup_still_has_duplicates", "activity_id has duplicates after dedup", dup_dedup)

    # ---------- Join sanity (dedup ⊆ filtered on activity_id) ----------
    try:
        only_in_dedup = con.sql(f"""
            WITH f AS (SELECT DISTINCT activity_id FROM read_parquet('{filtered.as_posix()}')),
                 d AS (SELECT DISTINCT activity_id FROM read_parquet('{dedup.as_posix()}'))
            SELECT d.activity_id FROM d LEFT JOIN f USING (activity_id) WHERE f.activity_id IS NULL
            LIMIT {args.sample_n}
        """).df()
        if len(only_in_dedup):
            record_fail("dedup_ids_not_in_filtered", f"{len(only_in_dedup)} dedup IDs not present in filtered", only_in_dedup)
    except Exception:
        # If activity_id was constructed differently, skip this
        print("Note: could not compare activity_id sets between filtered and dedup; skipping.")

    # ---------- Save metrics ----------
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics written -> {outdir/'metrics.json'}")

    if failures:
        (outdir / "failures.json").write_text(json.dumps(failures, indent=2))
        print(f"Found {len(failures)} issue(s). Details & samples saved in {outdir}/")
        if args.fail_on_error:
            sys.exit(1)
    else:
        print("All checks passed ✅")

    con.close()

if __name__ == "__main__":
    main()
