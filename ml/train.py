from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import joblib
import shap
import pandas as pd
import os
from ml.metrics import evaluate_classification_metrics


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='_distutils_hack')

def train_log_and_shap_classification(
    X_train, y_train, X_val, y_val, preprocessor,
    save_dir="saved_models", shap_dir="shap_outputs"
):
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
            'params': {'C': [0.1, 1.0, 10.0]}
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'params': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        },
        'XGBoost': {
            'model': XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        },
        'LightGBM': {
            'model': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        }
    }

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("LeadScoring_Simplified")

    results = []
    best_models = {}

    for name, model_info in models.items():
        print(f"\n🔧 Training: {name}")

        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model_info['model'])
        ])

        param_grid = {f"model__{k}": v for k, v in model_info['params'].items()}
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)

        y_val_pred = search.predict(X_val)
        y_val_proba = search.predict_proba(X_val)[:, 1] if hasattr(search.best_estimator_.named_steps['model'], "predict_proba") else None

        metrics = evaluate_classification_metrics(y_val, y_val_pred, y_val_proba)
        results.append({"model": name, "best_params": search.best_params_, **metrics})
        best_models[name] = search.best_estimator_

        model_path = os.path.join(save_dir, f"{name}_best_model.pkl")
        joblib.dump(search.best_estimator_, model_path)

        with mlflow.start_run(run_name=name):
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, "model")

            try:
                print(f"🔎 Generating SHAP values for {name}...")
                fitted_preprocessor = search.best_estimator_.named_steps['preprocess']
                X_val_proc = fitted_preprocessor.transform(X_val)
                shap_matrix = X_val_proc.toarray() if hasattr(X_val_proc, "toarray") else X_val_proc
                model_only = search.best_estimator_.named_steps['model']
                if name in ("RandomForest", "XGBoost", "LightGBM", "DecisionTree"):
                    explainer = shap.TreeExplainer(model_only)
                else:
                    explainer = shap.Explainer(model_only, shap_matrix)
                shap_values = explainer(shap_matrix)
                shap_path = os.path.join(shap_dir, f"{name}_shap_summary.png")
                plt.figure()
                shap.summary_plot(shap_values, shap_matrix, show=False)
                plt.savefig(shap_path, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(shap_path, artifact_path="shap_plots")
                print(f"✅ SHAP saved & logged: {shap_path}")
            except Exception as e:
                print(f"⚠️ SHAP failed for {name}: {e}")

    results_df = pd.DataFrame(results)
    print("\n📊 All Model Validation Metrics:")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))

    return results_df, best_models
