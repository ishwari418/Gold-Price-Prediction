"""Higher-level training runner that composes preprocessing, features and model training."""
import argparse
import joblib
import os
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import fit_transform_preprocessor
from src.features import add_features
from src.models import train_and_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='models/')
    parser.add_argument('--preproc', type=str, default='models/preprocessor.joblib')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = add_features(df)
    os.makedirs(args.output, exist_ok=True)
    X_trans, y, preprocessor = fit_transform_preprocessor(df, save_path=args.preproc)
    results = train_and_log(X_trans, y, preprocessor=preprocessor, out_dir=args.output)
    print('Done. Results:', results)


if __name__ == '__main__':
    main()
