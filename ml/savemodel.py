import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient

def save_and_register_best_model_pipeline(
    results_df, best_models, X_train_val, y_train_val, preprocessor,
    save_dir="saved_models", experiment_name="LeadScoring_Simplified"
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Select best model
    best_row = results_df.sort_values(by="f1_score", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = best_models[best_model_name]
    print(f"\n🏆 Best model selected: {best_model_name} (F1 = {best_row['f1_score']:.4f})")

    # 2. Final pipeline
    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", best_model.named_steps['model'] if hasattr(best_model, 'named_steps') else best_model)
    ])
    final_pipeline.fit(X_train_val, y_train_val)

    # 3. Save final model pipeline locally
    model_path = os.path.join(save_dir, f"final_{best_model_name}_pipeline.pkl")
    joblib.dump(final_pipeline, model_path)
    print(f"✅ Final pipeline saved at: {model_path}")

    # 4. Save preprocessor pipeline locally
    preprocessor_path = os.path.join(save_dir, "final_preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ Preprocessing pipeline saved at: {preprocessor_path}")

    # 5. Register to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"Final_{best_model_name}") as run:
        run_id = run.info.run_id

        # (a) Save local artifacts to MLflow
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessing_artifacts")

        # (b) Log full pipeline as MLflow model (for loading later)
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="sklearn_model",
            registered_model_name=best_model_name
        )

        print(f"🔁 Registering model to Model Registry: {best_model_name}")
        # --- Optionally: Get latest version programmatically (recommended, not just version=1)
        try:
            # List all versions in "None" stage (just logged)
            versions = client.get_latest_versions(name=best_model_name, stages=["None"])
            if versions:
                version = versions[0].version
                client.transition_model_version_stage(
                    name=best_model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True
                )
                print(f"✅ Model '{best_model_name}' v{version} moved to 'Production'.")
            else:
                print(f"⚠️ Could not find model version to move to 'Production'.")
        except Exception as e:
            print(f"⚠️ Transition to 'Production' failed: {e}")

        print(f"🏃 View run: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")
