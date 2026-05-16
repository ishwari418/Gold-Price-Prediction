from flask import Flask, request, jsonify
import joblib
import os
import sys
import traceback
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import add_features

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Try to load the best model automatically
MODEL_PATH = os.environ.get('GOLD_MODEL_PATH', 'models/RandomForest.joblib')
GOLD_DATA_PATH = os.environ.get('GOLD_DATA_PATH', 'data/gold.csv')
EXPECTED_FEATURE_COLUMNS = [
    'weight_grams', 'purity_karat', 'color', 'finish', 'hallmark',
    'depth', 'table', 'x', 'y', 'z', 'certification'
]


def load_artifacts(path=MODEL_PATH):
    if os.path.exists(path):
        data = joblib.load(path)
        model = data.get('model') if isinstance(data, dict) else data
        preprocessor = data.get('preprocessor') if isinstance(data, dict) else None
        return model, preprocessor
    return None, None


def model_is_compatible(preprocessor):
    if preprocessor is None:
        return True
    if hasattr(preprocessor, 'feature_names_in_'):
        return all(col in preprocessor.feature_names_in_ for col in EXPECTED_FEATURE_COLUMNS)
    if hasattr(preprocessor, 'transformers_'):
        feature_names = []
        for _, _, cols in preprocessor.transformers_:
            if isinstance(cols, (list, tuple)):
                feature_names.extend(cols)
            elif isinstance(cols, str):
                feature_names.append(cols)
        if feature_names:
            return all(col in feature_names for col in EXPECTED_FEATURE_COLUMNS)
    return False


def retrain_gold_model():
    try:
        import pandas as pd
        from src.models import train_and_log

        if not os.path.exists(GOLD_DATA_PATH):
            raise FileNotFoundError(f'Gold training data not found: {GOLD_DATA_PATH}')

        df = pd.read_csv(GOLD_DATA_PATH)
        if 'price' not in df.columns:
            raise ValueError('Gold dataset must include a `price` column.')

        from src.preprocessing import fit_transform_preprocessor
        from src.features import add_features
        
        df = add_features(df)
        X_trans, y, preprocessor = fit_transform_preprocessor(df)
        
        results = train_and_log(X_trans, y, preprocessor=preprocessor, out_dir=os.path.dirname(MODEL_PATH) or 'models')
        print('Retrained models:', ', '.join(results.keys()))
        return results.get('RandomForest', {}).get('path')
    except Exception as exc:
        print('Failed to retrain gold model:', exc)
        traceback.print_exc()
        return None


model, preprocessor = load_artifacts()
if model is None or not model_is_compatible(preprocessor):
    print('Loaded model is missing or incompatible with gold features. Retraining from data/gold.csv...')
    retrain_gold_model()
    model, preprocessor = load_artifacts()


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No JSON payload provided'}), 400

        # Expect single record or list
        records = payload if isinstance(payload, list) else [payload]
        import pandas as pd
        df = pd.DataFrame(records)

        defaults = {
            'purity_karat': '18K',
            'finish': 'Polished',
            'color': 'Yellow',
            'certification': 'BIS',
            'hallmark': 'Yes'
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        df = add_features(df)

        if preprocessor is not None:
            X = preprocessor.transform(df)
        else:
            X = df.values

        preds = model.predict(X)
        # simple confidence proxy using stddev of ensemble if available
        conf = None
        try:
            if hasattr(model, 'estimators_'):
                preds_all = np.vstack([est.predict(X) for est in model.estimators_])
                conf = preds_all.std(axis=0).tolist()
        except Exception:
            conf = None

        results = []
        for i, p in enumerate(preds.tolist()):
            results.append({'prediction': float(p), 'confidence': float(conf[i]) if conf is not None else None})

        return jsonify({'results': results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('GOLD_API_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
