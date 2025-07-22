import pandas as pd
from ml.dataingestion import data_ingestion
##from data.data_preprocessing.pipeline_builder import build_full_pipeline
from ml.train import train_log_and_shap_classification
from ml.savemodel import save_and_register_best_model_pipeline
from ml.datadrift import generate_and_log_drift_reports
from ml.prediction import load_and_predict_from_registry_auto
from data.data_preprocessing.mapping import map_categorical_columns, mapping_transformer
from data.data_preprocessing.pipeline_builder import build_full_pipeline
# from imblearn.over_sampling import SMOTE

def run_lead_prediction_pipeline(
    csv_file_path=None,
    table_name=None,
    experiment_name="LeadScoring_Simplified",
    save_dir="saved_models",
    shap_dir="shap_outputs",
    drift_dir="drift_reports"
):
    # 1. Ingest
    df = data_ingestion(csv_file_path=csv_file_path, table_name="lead")
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("❌ Data ingestion failed: No DataFrame returned.")

    # 2. Drop columns (customize as needed)
    drop_cols = [
        'Prospect ID', 'Lead Number', 'Get updates on DM Content',
        'Receive More Updates About Our Courses', 'I agree to pay the amount through cheque',
        'Magazine', 'Update me on Supply Chain Content'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # 3. Build preprocessor
    preprocessor = build_full_pipeline(df)

    # 4. Train/Val/Test split
    from sklearn.model_selection import train_test_split
    target_col = "Converted"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # sm = SMOTE(random_state=42)
    # X_train, y_train = sm.fit_resample(X_train, y_train)


    # 5. Data drift reports
    feature_names = X_train.columns if hasattr(X_train, "columns") else None
    generate_and_log_drift_reports(
        X_train, X_val, X_test,
        feature_names=feature_names,
        output_dir=drift_dir,
        mlflow_uri="http://127.0.0.1:5000",
        experiment_name="Drift"
    )

    # 6. Model training & SHAP
    results_df, best_models = train_log_and_shap_classification(
    X_train, y_train, X_val, y_val, preprocessor,
    save_dir=save_dir, shap_dir=shap_dir
    )


    # 7. Save & Register best model and preprocessor
    X_train_val = pd.concat([X_train, X_val,X_test])
    y_train_val = pd.concat([y_train, y_val,y_test])
    save_and_register_best_model_pipeline(
        results_df, best_models, X_train_val, y_train_val, preprocessor,
        save_dir=save_dir, experiment_name=experiment_name
    )
    # will load latest model in Staging
    y_pred_df = load_and_predict_from_registry_auto(X_test, stage="Production")
    ##print(y_pred_df.head())
    
# if you later move it to Production:
# y_pred = load_and_predict_from_registry_auto(X_test_raw, stage="Production")


if __name__ == "__main__":
    run_lead_prediction_pipeline(csv_file_path="data/Lead Scoring.csv")

