
import argparse, pandas as pd
from bio_thesis.eval.metrics import classification_report
ap = argparse.ArgumentParser()
ap.add_argument('--pred'); ap.add_argument('--gold')
args = ap.parse_args()
pred = pd.read_csv(args.pred); gold = pd.read_csv(args.gold)
df = gold.merge(pred, on='id')
print(classification_report(df['y_true'], df['y_pred'], digits=3))
