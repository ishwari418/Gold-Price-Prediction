"""Preprocessing utilities: missing values, encoders, scalers, and pipeline saving."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor(df: pd.DataFrame):
    numeric_features = ['weight_grams','depth','table','x','y','z']
    categorical_features = [c for c in df.columns if c not in numeric_features + ['price']]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features


def fit_transform_preprocessor(df: pd.DataFrame, save_path=None):
    preprocessor, numeric_features, categorical_features = build_preprocessor(df)
    X = df.drop(columns=['price'])
    y = df['price']
    X_trans = preprocessor.fit_transform(X)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({'preprocessor': preprocessor, 'num_features': numeric_features, 'cat_features': categorical_features}, save_path)
    return X_trans, y, preprocessor
