"""Evaluation and visualization utilities."""
import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_heatmap(df, out_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def actual_vs_pred(y_true, y_pred, out_path):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--models', type=str, required=True)
    parser.add_argument('--out', type=str, default='reports/')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    correlation_heatmap(df.select_dtypes(include=['number']), os.path.join(args.out, 'correlation.png'))
    print('Saved correlation heatmap')
