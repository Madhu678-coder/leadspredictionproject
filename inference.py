import joblib
import os
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    # Files are at root inside model.tar.gz, not inside a folder
    model_path = os.path.join(model_dir, "final_RandomForest_pipeline.pkl")
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
    return {
        "prediction": preds.tolist(),
        "probabilities": probs.tolist() if probs is not None else None
    }

