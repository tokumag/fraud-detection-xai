"""
Model training, evaluation, and persistence.

Supports XGBoost, Random Forest, and Logistic Regression with
comprehensive evaluation metrics for imbalanced classification.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(config: dict) -> object:
    """
    Build an untrained model from configuration.

    Parameters
    ----------
    config : dict
        Model section of config.yaml.

    Returns
    -------
    sklearn-compatible estimator
    """
    algorithm = config.get("algorithm", "xgboost")

    if algorithm == "xgboost":
        params = config.get("xgboost", {})
        scale_pos_weight = params.pop("scale_pos_weight", "auto")
        model = XGBClassifier(
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
            **{k: v for k, v in params.items() if k != "early_stopping_rounds"},
        )
        # Store for later use during fit
        model._early_stopping_rounds = params.get("early_stopping_rounds", 20)
        model._auto_scale = scale_pos_weight == "auto"

    elif algorithm == "random_forest":
        params = config.get("random_forest", {})
        model = RandomForestClassifier(random_state=42, **params)

    elif algorithm == "logistic_regression":
        params = config.get("logistic_regression", {})
        model = LogisticRegression(random_state=42, **params)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"✓ Built {algorithm} model")
    return model


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> object:
    """
    Train the model with optional early stopping (XGBoost).

    Parameters
    ----------
    model : estimator
        Untrained sklearn-compatible model.
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray, optional
        Validation data for early stopping.

    Returns
    -------
    Trained model.
    """
    if isinstance(model, XGBClassifier):
        # Auto-calculate scale_pos_weight
        if getattr(model, "_auto_scale", False):
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

        fit_params = {}
        if X_test is not None and y_test is not None:
            early_rounds = getattr(model, "_early_stopping_rounds", 20)
            model.set_params(early_stopping_rounds=early_rounds)
            fit_params["eval_set"] = [(X_test, y_test)]
            fit_params["verbose"] = False

        model.fit(X_train, y_train, **fit_params)
    else:
        model.fit(X_train, y_train)

    print("✓ Model training complete")
    return model


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """
    Evaluate the trained model with metrics suited for imbalanced classification.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  Specificity:{metrics['specificity']:.4f}")
    print("=" * 50)

    return metrics


def save_model(model, path: str) -> None:
    """Persist the trained model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"✓ Model saved to {path}")


def load_model(path: str):
    """Load a trained model from disk."""
    return joblib.load(path)


def save_metrics(metrics: dict, path: str) -> None:
    """Save evaluation metrics as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {path}")
