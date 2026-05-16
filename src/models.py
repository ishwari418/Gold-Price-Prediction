"""Model training, comparison and persistence."""
import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import mlflow
except ImportError:
    mlflow = None


def get_models(random_state=42):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state, objective='reg:squarederror', n_jobs=-1)
    }
    return models


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


def train_and_log(X, y, preprocessor=None, out_dir='models', random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    models = get_models(random_state)
    results = {}

    if mlflow is not None:
        mlflow.set_experiment('gold_price_prediction')

    for name, model in models.items():
        if mlflow is not None:
            mlflow_run = mlflow.start_run(run_name=name)
        else:
            mlflow_run = None

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            # Log metrics
            if mlflow is not None:
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
            # Save model
            model_path = os.path.join(out_dir, f"{name}.joblib")
            joblib.dump({'model': model, 'preprocessor': preprocessor}, model_path)
            if mlflow is not None:
                mlflow.log_artifact(model_path, artifact_path='models')
            results[name] = {'metrics': metrics, 'path': model_path}
        finally:
            if mlflow_run is not None:
                mlflow_run.end()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='models/')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    if 'price' not in df.columns:
        raise SystemExit('Input data must contain price column')
    X = df.drop(columns=['price'])
    y = df['price']
    # For training we expect preprocessor already applied in DVC pipeline; use raw X for now
    results = train_and_log(X, y, preprocessor=None, out_dir=args.output)
    print('Training completed. Models saved to', args.output)


if __name__ == '__main__':
    main()
