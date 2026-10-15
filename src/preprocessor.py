"""
Data preprocessing pipeline.

Handles feature scaling, train/test splitting, and class imbalance
using SMOTE or ADASYN oversampling.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN

from src.data_loader import FEATURE_NAMES, TARGET_NAME


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class Preprocessor:
    """
    End-to-end preprocessor for transaction data.

    Encapsulates scaling + resampling so the exact same transforms
    can be applied at inference time via ``transform()``.
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        handle_imbalance: str = "smote",
        test_size: float = 0.2,
        random_seed: int = 42,
    ):
        self.scaling_method = scaling_method
        self.handle_imbalance = handle_imbalance
        self.test_size = test_size
        self.random_seed = random_seed
        self.scaler = _SCALERS.get(scaling_method, StandardScaler)()
        self.feature_names = FEATURE_NAMES

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split, scale, and resample the training data.

        Returns
        -------
        X_train, X_test, y_train, y_test
            Processed arrays ready for model training.
        """
        X = df[self.feature_names].values
        y = df[TARGET_NAME].values

        # Train / test split (stratified to preserve fraud ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        # Fit scaler on training data only
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Handle class imbalance on training data only
        X_train, y_train = self._resample(X_train, y_train)

        print(f"✓ Preprocessing complete:")
        print(f"  Train: {X_train.shape[0]:,} samples ({y_train.sum():,.0f} fraud)")
        print(f"  Test:  {X_test.shape[0]:,} samples ({y_test.sum():,.0f} fraud)")

        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted scaler (for inference)."""
        X = df[self.feature_names].values
        return self.scaler.transform(X)

    def _resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply oversampling to balance classes."""
        if self.handle_imbalance == "smote":
            sampler = SMOTE(random_state=self.random_seed)
        elif self.handle_imbalance == "adasyn":
            sampler = ADASYN(random_state=self.random_seed)
        else:
            return X, y

        X_res, y_res = sampler.fit_resample(X, y)
        print(f"  Resampled with {self.handle_imbalance.upper()}: "
              f"{len(y):,} → {len(y_res):,} samples")
        return X_res, y_res

    def save(self, path: str) -> None:
        """Persist the fitted preprocessor."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✓ Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Load a previously fitted preprocessor."""
        return joblib.load(path)
