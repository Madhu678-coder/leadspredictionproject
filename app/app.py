from flask import Flask, render_template, request
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

app = Flask(__name__)

# ==============================
# ✅ Final 35 features
# ==============================
features = [
    'Lead Origin',
    'Lead Source',
    'Do Not Email',
    'Do Not Call',
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Last Activity',
    'Country',
    'Specialization',
    'How did you hear about X Education',
    'What is your current occupation',
    'What matters most to you in choosing a course',
    'Search',
    'Magazine',
    'Newspaper Article',
    'X Education Forums',
    'Newspaper',
    'Digital Advertisement',
    'Through Recommendations',
    'Receive More Updates About Our Courses',
    'Tags',
    'Lead Quality',
    'Update me on Supply Chain Content',
    'Get updates on DM Content',
    'Lead Profile',
    'City',
    'Asymmetrique Activity Index',
    'Asymmetrique Profile Index',
    'Asymmetrique Activity Score',
    'Asymmetrique Profile Score',
    'I agree to pay the amount through cheque',
    'A free copy of Mastering The Interview',
    'Last Notable Activity'
]

# ==============================
# ✅ Connect to MLflow and fetch the latest Production model
# ==============================
mlflow.set_tracking_uri("http://localhost:5000")  # change if needed

def get_latest_production_model():
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("❌ No models found in MLflow registry!")
    
    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if lv.current_stage == "Production":
                candidates.append((m.name, lv.version, lv.creation_timestamp))
    
    if not candidates:
        raise RuntimeError("❌ No model is currently in Production stage!")
    
    # Sort by latest creation timestamp
    candidates.sort(key=lambda x: x[2], reverse=True)
    model_name, version, _ = candidates[0]
    print(f"✅ Loading model: {model_name} (version {version}) from Production...")
    # ✅ Use sklearn flavor to get full model with predict_proba
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")

# Load model at startup
try:
    pipeline = get_latest_production_model()
    print("✅ Model loaded successfully and ready for predictions!")
except Exception as e:
    print(f"❌ Failed to load model from MLflow: {e}")
    pipeline = None

# ==============================
# ✅ Routes
# ==============================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return render_template("predict.html", prediction_text="❌ No model available. Please check MLflow registry.")

    try:
        # Build input data from form
        input_data = {feat: request.form.get(feat, '') for feat in features}

        # Convert numeric fields
        numeric_cols = [
            'TotalVisits',
            'Total Time Spent on Website',
            'Page Views Per Visit',
            'Asymmetrique Activity Index',
            'Asymmetrique Profile Index',
            'Asymmetrique Activity Score',
            'Asymmetrique Profile Score'
        ]
        for col in numeric_cols:
            val = input_data.get(col, '')
            input_data[col] = float(val) if val not in (None, '', 'None') else 0.0

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # ✅ Predict using sklearn model
        prediction = int(pipeline.predict(input_df)[0])
        proba = pipeline.predict_proba(input_df)[0]
        confidence = float(max(proba))

        # ✅ Human-readable output
        if prediction == 1:
            if confidence >= 0.7:
                chance = 'High <span style="color:green;">&#128161;</span>'
            elif confidence >= 0.4:
                chance = 'Medium <span style="color:orange;">&#128528;</span>'
            else:
                chance = 'Low <span style="color:#c0392b;">&#9888;&#65039;</span>'
            prediction_text = (
                f"The chance that this person will convert is: "
                f"<b>{chance}</b> <br>(probability: {confidence:.1%})"
            )
        else:
            prediction_text = "This lead is predicted <b>not likely to convert</b>."

    except Exception as e:
        prediction_text = f"❌ Error: {str(e)}"

    return render_template("predict.html", prediction_text=prediction_text)

# ==============================
# ✅ Main
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
