from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import logging
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# -------------------------------
# 🔧 Flask setup
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# 🔧 MLflow tracking (replace with your own URI)
# -------------------------------
mlflow.set_tracking_uri("arn:aws:sagemaker:ap-south-1:888517279277:mlflow-tracking-server/capstone")

def get_latest_production_model_name(stage="Production", alias=None):
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("❌ No models registered in MLflow!")

    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if alias:
                aliases = getattr(lv, 'aliases', [])
                if alias in aliases:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))
            else:
                if lv.current_stage == stage:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))

    if not candidates:
        raise ValueError(f"❌ No model found for stage='{stage}' alias='{alias}'")

    candidates.sort(key=lambda t: t[2], reverse=True)
    chosen_model = candidates[0][0]
    logger.info(f"✅ Will load model: {chosen_model} (version {candidates[0][1]})")
    return chosen_model

def load_model_from_registry(stage="Production", alias=None):
    try:
        model_name = get_latest_production_model_name(stage=stage, alias=alias)
        model_uri = f"models:/{model_name}/{alias or stage}"
        logger.info(f"📦 Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully from MLflow")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from MLflow: {str(e)}")
        raise

# Load pipeline at app startup
pipeline = load_model_from_registry(stage="Production", alias=None)

# -------------------------------
# 🌐 Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index_bulk.html')

@app.route('/predict', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "❌ No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "❌ No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)

        # ✅ Run pipeline
        preds = pipeline.predict(df)

        # ✅ Add prediction columns
        df['Prediction'] = preds
        df['Prediction_Label'] = ["✅ Lead will convert" if p == 1 else "❌ Lead will not convert" for p in preds]

        # ✅ Save results
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"scored_{filename}")
        df.to_csv(output_path, index=False)

        return jsonify({
            "message": "✅ Prediction successful!",
            "data": df.head(100).to_dict(orient='records'),  # send ALL columns
            "download_path": output_path
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"❌ Prediction failed: {str(e)}"}), 500

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

# -------------------------------
# 🚀 Run with ngrok
# -------------------------------
if __name__ == "__main__":
    port = 8081
    NGROK_AUTH_TOKEN = "308Pu4UuOlqGRn3NubKofC6fgFq_7f9Ayx9MFUxYThGu2Lr8v"
    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        http_tunnel = ngrok.connect(port)
        logger.info(f"🌐 Public URL: {http_tunnel.public_url}")
    except Exception as e:
        logger.error(f"❌ Failed to start ngrok tunnel: {str(e)}")

    app.run(host="0.0.0.0", port=port, debug=True)
