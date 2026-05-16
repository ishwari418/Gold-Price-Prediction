"""Feature engineering utilities for gold pricing."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # volume proxy for jewelry or bullion dimensions
    df['volume'] = df['x'] * df['y'] * df['z']
    # dimensions ratios
    df['xy_ratio'] = df['x'] / df['y'].replace(0, 1)
    df['depth_percent'] = df['depth']
    # interaction feature for gold weight and shape
    df['weight_volume'] = df['weight_grams'] * df['volume']
    return df
