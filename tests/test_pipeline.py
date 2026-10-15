"""
Unit tests for the fraud detection pipeline.

Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, FEATURE_NAMES, TARGET_NAME
from src.preprocessor import Preprocessor
from src.model import build_model, train_model, evaluate_model


@pytest.fixture
def sample_data():
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_data(n_samples=2000, fraud_ratio=0.05, random_seed=42)


@pytest.fixture
def config():
    """Minimal model configuration for testing."""
    return {
        "algorithm": "xgboost",
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": "auto",
            "eval_metric": "aucpr",
            "early_stopping_rounds": 5,
        },
    }


class TestDataLoader:
    """Tests for data generation."""

    def test_generate_correct_shape(self, sample_data):
        assert sample_data.shape[0] == 2000
        assert sample_data.shape[1] == len(FEATURE_NAMES) + 1

    def test_correct_columns(self, sample_data):
        expected_cols = set(FEATURE_NAMES + [TARGET_NAME])
        assert set(sample_data.columns) == expected_cols

    def test_fraud_ratio(self, sample_data):
        fraud_ratio = sample_data[TARGET_NAME].mean()
        assert 0.03 <= fraud_ratio <= 0.07  # ~5% ± tolerance

    def test_no_null_values(self, sample_data):
        assert sample_data.isnull().sum().sum() == 0

    def test_target_binary(self, sample_data):
        assert set(sample_data[TARGET_NAME].unique()) == {0, 1}


class TestPreprocessor:
    """Tests for preprocessing pipeline."""

    def test_fit_transform_shapes(self, sample_data):
        prep = Preprocessor(handle_imbalance="none", test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test = prep.fit_transform(sample_data)

        assert X_test.shape[0] == int(2000 * 0.2)
        assert X_train.shape[1] == len(FEATURE_NAMES)
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_scaling_applied(self, sample_data):
        prep = Preprocessor(handle_imbalance="none", random_seed=42)
        X_train, X_test, _, _ = prep.fit_transform(sample_data)

        # Scaled data should have mean ~0 and std ~1 for training set
        assert np.abs(X_train.mean()) < 0.5
        assert np.abs(X_train.std() - 1.0) < 0.5

    def test_smote_increases_minority(self, sample_data):
        prep = Preprocessor(handle_imbalance="smote", random_seed=42)
        X_train, _, y_train, _ = prep.fit_transform(sample_data)

        # After SMOTE, classes should be balanced
        fraud_ratio = y_train.mean()
        assert 0.4 <= fraud_ratio <= 0.6


class TestModel:
    """Tests for model training and evaluation."""

    def test_build_model(self, config):
        model = build_model(config)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_train_and_predict(self, sample_data, config):
        prep = Preprocessor(handle_imbalance="smote", test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test = prep.fit_transform(sample_data)

        model = build_model(config)
        model = train_model(model, X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    def test_evaluate_returns_metrics(self, sample_data, config):
        prep = Preprocessor(handle_imbalance="smote", test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test = prep.fit_transform(sample_data)

        model = build_model(config)
        model = train_model(model, X_train, y_train, X_test, y_test)

        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1

    def test_model_performance_baseline(self, sample_data, config):
        """Ensure model beats random baseline."""
        prep = Preprocessor(handle_imbalance="smote", test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test = prep.fit_transform(sample_data)

        model = build_model(config)
        model = train_model(model, X_train, y_train, X_test, y_test)
        metrics = evaluate_model(model, X_test, y_test)

        # Model should significantly beat random (0.5 ROC-AUC)
        assert metrics["roc_auc"] > 0.7
