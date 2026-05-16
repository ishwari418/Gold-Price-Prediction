"""Generate a synthetic gold pricing dataset for development and testing."""
import argparse
import os
import numpy as np
import pandas as pd


def generate(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    weight_grams = np.round(rng.uniform(5.0, 200.0, size=n), 1)
    purity_karat = rng.choice(['10K', '14K', '18K', '22K', '24K'], size=n, p=[0.05, 0.2, 0.35, 0.25, 0.15])
    color = rng.choice(['Yellow', 'White', 'Rose'], size=n, p=[0.5, 0.3, 0.2])
    finish = rng.choice(['Polished', 'Matte', 'Hammered', 'Satin'], size=n, p=[0.5, 0.2, 0.15, 0.15])
    hallmark = rng.choice(['Yes', 'No'], size=n, p=[0.7, 0.3])
    depth = np.round(rng.uniform(10, 55, size=n), 2)
    table = np.round(rng.uniform(40, 80, size=n), 2)
    x = np.round(weight_grams ** (1/3) * rng.uniform(1.5, 3.5, size=n), 2)
    y = np.round(x * rng.uniform(0.9, 1.1, size=n), 2)
    z = np.round(depth / 100 * ((x + y) / 2), 2)
    certification = rng.choice(['BIS', 'LBMA', 'IGI', 'None'], size=n, p=[0.5, 0.2, 0.1, 0.2])

    # synthetic price signal for gold
    base_price = weight_grams * 60
    purity_score = pd.Series(purity_karat).map({'10K':0.7, '14K':0.9, '18K':1.1, '22K':1.25, '24K':1.4}).values
    color_score = pd.Series(color).map({'Yellow':1.0, 'White':1.05, 'Rose':1.03}).values
    finish_score = pd.Series(finish).map({'Polished':1.0, 'Matte':0.98, 'Hammered':1.02, 'Satin':1.01}).values
    hallmark_score = pd.Series(hallmark).map({'Yes':1.1, 'No':0.95}).values
    cert_score = pd.Series(certification).map({'BIS':1.08, 'LBMA':1.06, 'IGI':1.04, 'None':0.95}).values
    shape_score = np.clip(1 + (depth - 35) * 0.002 + (table - 60) * 0.0015, 0.85, 1.2)

    noise = rng.normal(0, 100, size=n)
    price = np.round(base_price * purity_score * color_score * finish_score * hallmark_score * cert_score * shape_score + noise)

    df = pd.DataFrame({
        'weight_grams': weight_grams,
        'purity_karat': purity_karat,
        'color': color,
        'finish': finish,
        'hallmark': hallmark,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
        'certification': certification,
        'price': price
    })
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data/gold.csv')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = generate(n=args.n, random_state=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote synthetic dataset to {args.output} ({len(df)} rows)")


if __name__ == '__main__':
    main()
