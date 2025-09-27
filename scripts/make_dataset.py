
import argparse, pandas as pd
from bio_thesis.data.load_crs import read_crs_parquet, save_parquet
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inp', required=True)
ap.add_argument('--out', dest='out', required=True)
ap.add_argument('--columns', nargs='*')
ap.add_argument('--sample', type=int, default=0)
args = ap.parse_args()
df = read_crs_parquet(args.inp, columns=args.columns)
if args.sample and args.sample < len(df): df = df.sample(args.sample, random_state=42)
save_parquet(df, args.out); print(f'Saved {len(df)} rows to {args.out}')
