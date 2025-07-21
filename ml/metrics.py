from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification_metrics(y_true, y_pred, y_proba=None, average_type='binary'):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average_type, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average_type, zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics

def print_classification_report(y_true, y_pred):
    print("\n✅ Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
