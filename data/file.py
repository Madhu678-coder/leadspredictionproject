import joblib
model = joblib.load("saved_models/final_RandomForest_pipeline.pkl")
print(hasattr(model, "predict_proba"))