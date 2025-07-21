import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

def get_latest_model_name(stage="Production", alias=None):
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("❌ No models registered in MLflow!")

    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if alias:
                # Check aliases if provided
                aliases = getattr(lv, "aliases", [])
                if alias in aliases:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))
            else:
                # Check stage
                if lv.current_stage == stage:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))

    if not candidates:
        raise ValueError(f"❌ No model found in MLflow registry for stage='{stage}' alias='{alias}'")

    candidates.sort(key=lambda t: t[2], reverse=True)
    chosen_model, chosen_version, _ = candidates[0]
    print(f"✅ Will load {chosen_model} version {chosen_version} (stage/alias: '{alias or stage}')")
    return chosen_model

def map_confidence(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def load_and_predict_from_registry_auto(X_test, stage="Production", alias=None):
    model_name = get_latest_model_name(stage=stage, alias=alias)
    model_uri = f"models:/{model_name}/{alias or stage}"
    print(f"📦 Loading model from: {model_uri}")
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)

    if hasattr(loaded_pipeline, "predict_proba"):
        proba = loaded_pipeline.predict_proba(X_test)
        pos_probs = proba[:, 1]  # probability for positive class
        predictions = ["Convert" if p >= 0.5 else "Not Convert" for p in pos_probs]
        confidences = [map_confidence(p) for p in pos_probs]

        result_df = pd.DataFrame({
            "Prediction": predictions,
            "Probability": pos_probs,
            "Confidence": confidences
        })
        print(f"✅ Predictions with confidence ready. Example:\n{result_df.head()}")
        return result_df
    else:
        preds = loaded_pipeline.predict(X_test)
        result_df = pd.DataFrame({
            "Prediction": preds,
            "Probability": ["N/A"] * len(preds),
            "Confidence": ["N/A"] * len(preds)
        })
        print("⚠️ Model does not support predict_proba. Returning class predictions only.")
        return result_df
