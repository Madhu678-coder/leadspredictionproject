from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# ✅ PostgreSQL config
# -----------------------------
DB_USER = "postgres"
DB_PASSWORD = "Madhu14777"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydb5"

# -----------------------------
# ✅ MLflow tracking URI
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")

def get_latest_staging_model():
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("❌ No models registered in MLflow!")

    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if lv.current_stage == "Production":
                candidates.append((m.name, lv.version, lv.creation_timestamp))

    if not candidates:
        raise RuntimeError("❌ No model found in Production stage!")

    candidates.sort(key=lambda x: x[2], reverse=True)
    model_name, version, _ = candidates[0]
    print(f"✅ Loading model: {model_name} (version {version}) from Production...")
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# 🔄 Load the model ONCE at app startup
try:
    model_pipeline = get_latest_staging_model()
    print("✅ Model loaded successfully and ready for predictions!")
except Exception as e:
    print(f"❌ Failed to load model from MLflow: {e}")
    model_pipeline = None

# -----------------------------
# Helper: map probability to High/Medium/Low
# (not used anymore but kept for reference)
# -----------------------------
def map_confidence(prob):
    if prob is None or np.isnan(prob):
        return None
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def index():
    return render_template('index_bulk.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({"error": "Model not available. Please check MLflow registry."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        # Drop columns not needed
        df.drop(columns=['prospect_id', 'lead_number'], inplace=True, errors='ignore')
        # Mark blanks as NaN
        df.replace(["Select", "", "None"], np.nan, inplace=True)
        df = df.fillna(0)
    except Exception as e:
        return jsonify({"error": f"Failed to read/clean CSV: {str(e)}"}), 500

    try:
        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(df)
            if isinstance(proba, pd.DataFrame):
                pos_probs = proba.iloc[:, 1].values
            else:
                pos_probs = proba[:, 1]
            predictions = ["Convert" if p >= 0.5 else "Not Convert" for p in pos_probs]
            df['Prediction'] = predictions
            df['Probability'] = pos_probs

            # --- REMOVE Confidence column! ---
            # confidence_levels = [map_confidence(p) for p in pos_probs]
            # df['Confidence'] = confidence_levels

        else:
            preds = model_pipeline.predict(df)
            df['Prediction'] = preds
            df['Probability'] = None
            # df['Confidence'] = None

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    print(df.head())  # Debug to confirm prediction/probability present

    df = df.replace({np.nan: None})

    try:
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        df_to_save = df.drop(columns=['Prediction', 'Probability'], errors='ignore')
        df_to_save.to_sql("lead_scoring", engine, if_exists="append", index=False)
    except Exception as e:
        return jsonify({"error": f"Failed to save to DB: {str(e)}"}), 500

    # ---- REMOVE Confidence from return! ----
    df.drop(columns=['Confidence'], inplace=True, errors='ignore')
    return jsonify({
        "message": "Prediction successful",
        "data": df.head(100).to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
