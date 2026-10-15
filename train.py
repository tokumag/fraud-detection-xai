#!/usr/bin/env python3
"""
Training pipeline entry point.

Usage:
    python train.py                    # Use default config.yaml
    python train.py --config custom.yaml
"""

import argparse
import yaml
from pathlib import Path

from src.data_loader import generate_synthetic_data, load_data
from src.preprocessor import Preprocessor
from src.model import build_model, train_model, evaluate_model, save_model, save_metrics
from src.explainer import FraudExplainer


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml") -> None:
    """Run the full training pipeline."""
    config = load_config(config_path)
    print("=" * 60)
    print(f"  FRAUD DETECTION XAI — Training Pipeline v{config['project']['version']}")
    print("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────
    print("\n[1/5] Generating synthetic transaction data...")
    data_cfg = config["data"]
    data_path = data_cfg["output_path"]

    if Path(data_path).exists():
        print(f"  Loading existing dataset from {data_path}")
        df = load_data(data_path)
    else:
        df = generate_synthetic_data(
            n_samples=data_cfg["n_samples"],
            fraud_ratio=data_cfg["fraud_ratio"],
            random_seed=config["project"]["random_seed"],
            save_path=data_path,
        )

    print(f"  Dataset shape: {df.shape}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

    # ── Step 2: Preprocessing ────────────────────────────────────
    print("\n[2/5] Preprocessing data...")
    prep_cfg = config["preprocessing"]
    preprocessor = Preprocessor(
        scaling_method=prep_cfg["scaling_method"],
        handle_imbalance=prep_cfg["handle_imbalance"],
        test_size=prep_cfg["test_size"],
        random_seed=config["project"]["random_seed"],
    )
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # ── Step 3: Model Training ───────────────────────────────────
    print("\n[3/5] Training model...")
    model = build_model(config["model"])
    model = train_model(model, X_train, y_train, X_test, y_test)

    # ── Step 4: Evaluation ───────────────────────────────────────
    print("\n[4/5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # ── Step 5: Explainability ───────────────────────────────────
    print("\n[5/5] Generating SHAP explanations...")
    explainer_cfg = config["explainability"]
    explainer = FraudExplainer(
        model, X_train, max_samples=explainer_cfg["shap_max_samples"]
    )

    # Global feature importance
    importance_df = explainer.global_feature_importance(
        X_test[:explainer_cfg["shap_max_samples"]],
        top_k=explainer_cfg["top_k_features"],
    )
    print("\nTop features driving fraud detection:")
    print(importance_df.to_string(index=False))

    # Generate plots
    if explainer_cfg.get("generate_plots", True):
        explainer.plot_summary(
            X_test[:explainer_cfg["shap_max_samples"]],
            save_path="models/shap_summary.png",
        )
        explainer.plot_bar(
            X_test[:explainer_cfg["shap_max_samples"]],
            top_k=explainer_cfg["top_k_features"],
            save_path="models/shap_bar.png",
        )

    # ── Save Artifacts ───────────────────────────────────────────
    artifacts = config["artifacts"]
    save_model(model, artifacts["model_path"])
    preprocessor.save(artifacts["preprocessor_path"])
    save_metrics(metrics, artifacts["metrics_path"])

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete! All artifacts saved.")
    print("  Run `streamlit run app.py` to launch the demo.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
