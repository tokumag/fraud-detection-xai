"""
SHAP-based model explainability module.

Generates global and local explanations for fraud detection predictions,
including feature importance, summary plots, and per-transaction explanations.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import FEATURE_NAMES


class FraudExplainer:
    """
    Wrapper around SHAP for explaining fraud detection models.

    Supports both TreeExplainer (XGBoost, RF) and KernelExplainer (LR)
    with automatic backend selection.
    """

    def __init__(self, model, X_background: np.ndarray, max_samples: int = 500):
        """
        Parameters
        ----------
        model : trained estimator
            The fraud detection model.
        X_background : np.ndarray
            Background data for SHAP (subset of training data).
        max_samples : int
            Max samples to use for background data.
        """
        self.model = model
        self.feature_names = FEATURE_NAMES

        # Subsample background data for efficiency
        if len(X_background) > max_samples:
            idx = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[idx]

        # Auto-select explainer based on model type
        model_type = type(model).__name__
        if model_type in ("XGBClassifier", "RandomForestClassifier"):
            self.explainer = shap.TreeExplainer(model)
            self._explainer_type = "tree"
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba, X_background
            )
            self._explainer_type = "kernel"

        print(f"✓ SHAP {self._explainer_type} explainer initialized")

    def compute_shap_values(self, X: np.ndarray) -> shap.Explanation:
        """
        Compute SHAP values for the given samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to explain (scaled features).

        Returns
        -------
        shap.Explanation
            SHAP values with feature names attached.
        """
        shap_values = self.explainer(X)

        # For binary classifiers, take the fraud class (index 1)
        if isinstance(shap_values.values, list):
            shap_values = shap_values[..., 1]
        elif shap_values.values.ndim == 3:
            shap_values = shap_values[..., 1]

        shap_values.feature_names = self.feature_names
        return shap_values

    def global_feature_importance(
        self, X: np.ndarray, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Compute mean absolute SHAP values (global feature importance).

        Returns
        -------
        pd.DataFrame
            Sorted feature importance table.
        """
        shap_values = self.compute_shap_values(X)
        importance = np.abs(shap_values.values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False).head(top_k)
        df = df.reset_index(drop=True)
        return df

    def explain_single(self, X_single: np.ndarray) -> dict:
        """
        Explain a single transaction prediction.

        Parameters
        ----------
        X_single : np.ndarray
            Single sample (1, n_features).

        Returns
        -------
        dict
            Feature contributions for this prediction.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.compute_shap_values(X_single)
        contributions = dict(zip(self.feature_names, shap_values.values[0]))

        # Sort by absolute contribution
        contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return contributions

    # --- Plot generation ---------------------------------------------------

    def plot_summary(
        self, X: np.ndarray, save_path: str | None = None
    ) -> plt.Figure:
        """Generate SHAP beeswarm summary plot."""
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("SHAP Feature Impact on Fraud Prediction", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Summary plot saved to {save_path}")

        return fig

    def plot_bar(
        self, X: np.ndarray, top_k: int = 10, save_path: str | None = None
    ) -> plt.Figure:
        """Generate SHAP bar plot of mean feature importance."""
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=top_k, show=False)
        plt.title("Top Features Driving Fraud Detection", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Bar plot saved to {save_path}")

        return fig

    def plot_waterfall(
        self, X_single: np.ndarray, index: int = 0, save_path: str | None = None
    ) -> plt.Figure:
        """Generate SHAP waterfall plot for a single prediction."""
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.compute_shap_values(X_single)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[index], show=False)
        plt.title("Feature Contributions to This Prediction", fontsize=14)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Waterfall plot saved to {save_path}")

        return fig
