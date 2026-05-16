"""Simple CLI wrapper to preprocess and emit processed CSV for pipeline."""
import argparse
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import add_features
from src.preprocessing import fit_transform_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/processed.csv')
    parser.add_argument('--preproc', type=str, default='models/preprocessor.joblib')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = add_features(df)
    X_trans, y, preprocessor = fit_transform_preprocessor(df, save_path=args.preproc)
    # store a processed CSV with original features plus engineered ones
    df.to_csv(args.output, index=False)
    print('Wrote processed data to', args.output)


if __name__ == '__main__':
    main()
