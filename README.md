# Gold Price Prediction & Quality Analysis

This repository is a full-stack AI-powered gold valuation and quality analysis platform showcasing an end-to-end ML workflow with DVC and MLflow integration, a Flask API, and a Tailwind-powered frontend dashboard.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Generate a synthetic dataset (if you don't have one):

```bash
py scripts/generate_synthetic_data.py --output data/gold.csv --n 5000
```

3. Run the training pipeline (uses MLflow for logging):

```bash
py src/train.py --data data/gold.csv --output models/
```

4. Run the API:

```bash
py app.py
```

By default, the Flask API will start on port `5000` to avoid conflicts with services like Splunk on `8000`.

5. Open the frontend: `frontend/index.html` (or serve it with a static server).

DVC and MLflow

- `dvc.yaml` contains pipeline stages for data generation, preprocessing, training, and evaluation.
- Use `dvc add data/gold.csv` and `dvc push` to version datasets.
- Start MLflow UI: `mlflow ui --port 5001` to inspect experiments.

Project structure

- `src/` - preprocessing, training, models and evaluation modules
- `scripts/` - dataset generation utilities
- `models/` - saved models and preprocessing pipelines
- `frontend/` - HTML/CSS/JS dashboard
- `app.py` - Flask prediction API
