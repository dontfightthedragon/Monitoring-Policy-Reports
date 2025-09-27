
import argparse, pandas as pd, joblib
from bio_thesis.models.topic_model import run_bertopic
ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='inp', required=True)
ap.add_argument('--text-col', default='DESCRIPTION'); ap.add_argument('--out', dest='out', required=True)
args = ap.parse_args()
df = pd.read_parquet(args.inp) if args.inp.endswith('.parquet') else pd.read_csv(args.inp)
model = run_bertopic(df[args.text_col].astype(str).tolist())
joblib.dump(model, args.out + '/bertopic.joblib'); print('Model saved')
