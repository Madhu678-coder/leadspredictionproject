import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_and_log_drift_reports(
    X_train, X_val, X_test,
    feature_names=None,
    output_dir="drift_reports",
    mlflow_uri="http://127.0.0.1:5000",
    experiment_name="Drift"
):
    """
    Generates Evidently Data Drift reports comparing train/val/test,
    saves them as HTML, and logs both artifacts and metrics into MLflow.

    Returns
    -------
    dict
        A dictionary with drift metrics for each comparison.
    """

    # Helper to ensure DataFrame
    def ensure_df(data, feature_names):
        if isinstance(data, pd.DataFrame):
            return data
        cols = feature_names if feature_names is not None else [f"feature_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols)

    X_train = ensure_df(X_train, feature_names)
    X_val = ensure_df(X_val, feature_names)
    X_test = ensure_df(X_test, feature_names)

    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    comparisons = [
        ("train_vs_val", X_train, X_val),
        ("train_vs_test", X_train, X_test),
        ("val_vs_test", X_val, X_test)
    ]

    results_summary = {}

    with mlflow.start_run(run_name="multi_split_drift") as run:
        for name, ref, curr in comparisons:
            print(f"🚀 Running drift check: {name}")
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref, current_data=curr)

            # Save HTML artifact
            html_path = os.path.join(output_dir, f"{name}.html")
            report.save_html(html_path)
            mlflow.log_artifact(html_path, artifact_path="evidently_html_reports")

            json_dict = report.as_dict()
            drift_result = next((m["result"] for m in json_dict["metrics"] if m.get("metric") == "DataDriftTable"), None)

            if drift_result:
                drift_ratio = drift_result.get("share_of_drifted_columns", 0)
                mlflow.log_metric(f"{name}_drift_ratio", round(drift_ratio, 4))

                column_metrics = {}
                for feature, vals in drift_result.get("drift_by_columns", {}).items():
                    score = vals.get("drift_score")
                    if score is not None:
                        clean_name = feature.replace(" ", "_").replace("(", "").replace(")", "")
                        mlflow.log_metric(f"{name}_{clean_name}", round(score, 4))
                        column_metrics[feature] = round(score, 4)

                results_summary[name] = {
                    "drift_ratio": round(drift_ratio, 4),
                    "column_scores": column_metrics
                }

            print(f"✅ Drift metrics for {name} logged to MLflow.\n")

        print(f"🎯 Drift reports & metrics logged under run ID: {run.info.run_id}")

    return results_summary
