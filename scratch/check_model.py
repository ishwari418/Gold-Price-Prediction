import joblib
import os

path = 'models/RandomForest.joblib'
if os.path.exists(path):
    data = joblib.load(path)
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        print(f"Model: {type(data.get('model'))}")
        print(f"Preprocessor: {type(data.get('preprocessor'))}")
    else:
        print(f"Not a dict, type: {type(data)}")
else:
    print("File not found")
