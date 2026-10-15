"""
Synthetic transaction data generator.

Generates realistic financial transaction data with configurable fraud ratio,
simulating patterns commonly seen in real-world fraud detection scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# --- Feature name constants -----------------------------------------------
FEATURE_NAMES = [
    "transaction_amount",
    "merchant_category",
    "transaction_hour",
    "transaction_day_of_week",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase",
    "repeat_retailer",
    "used_chip",
    "used_pin",
    "online_order",
    "account_age_days",
    "num_transactions_last_24h",
    "num_transactions_last_7d",
    "avg_amount_last_7d",
    "std_amount_last_7d",
    "max_amount_last_30d",
    "time_since_last_transaction",
    "is_weekend",
    "is_international",
]

TARGET_NAME = "is_fraud"


def generate_synthetic_data(
    n_samples: int = 50_000,
    fraud_ratio: float = 0.02,
    random_seed: int = 42,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic fraud detection dataset.

    The generator creates realistic patterns where:
    - Fraudulent transactions tend to be larger amounts
    - Fraud is more likely for online/international orders
    - Fraud occurs more frequently at unusual hours
    - Legitimate users tend to use chip & PIN

    Parameters
    ----------
    n_samples : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (0.0 to 1.0).
    random_seed : int
        Random seed for reproducibility.
    save_path : str, optional
        If provided, save the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Generated transaction dataset with features and target.
    """
    rng = np.random.RandomState(random_seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions -------------------------------------------
    legit = {
        "transaction_amount": rng.exponential(50, n_legit) + 1,
        "merchant_category": rng.randint(0, 15, n_legit),
        "transaction_hour": rng.normal(14, 4, n_legit).clip(0, 23).astype(int),
        "transaction_day_of_week": rng.randint(0, 7, n_legit),
        "distance_from_home": rng.exponential(10, n_legit),
        "distance_from_last_transaction": rng.exponential(5, n_legit),
        "ratio_to_median_purchase": rng.normal(1.0, 0.3, n_legit).clip(0.1),
        "repeat_retailer": rng.binomial(1, 0.8, n_legit),
        "used_chip": rng.binomial(1, 0.7, n_legit),
        "used_pin": rng.binomial(1, 0.6, n_legit),
        "online_order": rng.binomial(1, 0.3, n_legit),
        "account_age_days": rng.normal(800, 400, n_legit).clip(30),
        "num_transactions_last_24h": rng.poisson(3, n_legit),
        "num_transactions_last_7d": rng.poisson(15, n_legit),
        "avg_amount_last_7d": rng.exponential(45, n_legit) + 5,
        "std_amount_last_7d": rng.exponential(20, n_legit),
        "max_amount_last_30d": rng.exponential(100, n_legit) + 10,
        "time_since_last_transaction": rng.exponential(12, n_legit),
        "is_weekend": rng.binomial(1, 2 / 7, n_legit),
        "is_international": rng.binomial(1, 0.05, n_legit),
    }

    # --- Fraudulent transactions -------------------------------------------
    fraud = {
        "transaction_amount": rng.exponential(200, n_fraud) + 50,
        "merchant_category": rng.randint(0, 15, n_fraud),
        "transaction_hour": rng.choice([0, 1, 2, 3, 4, 22, 23], n_fraud),
        "transaction_day_of_week": rng.randint(0, 7, n_fraud),
        "distance_from_home": rng.exponential(80, n_fraud),
        "distance_from_last_transaction": rng.exponential(50, n_fraud),
        "ratio_to_median_purchase": rng.normal(4.0, 2.0, n_fraud).clip(0.5),
        "repeat_retailer": rng.binomial(1, 0.2, n_fraud),
        "used_chip": rng.binomial(1, 0.2, n_fraud),
        "used_pin": rng.binomial(1, 0.15, n_fraud),
        "online_order": rng.binomial(1, 0.75, n_fraud),
        "account_age_days": rng.normal(200, 150, n_fraud).clip(1),
        "num_transactions_last_24h": rng.poisson(8, n_fraud),
        "num_transactions_last_7d": rng.poisson(25, n_fraud),
        "avg_amount_last_7d": rng.exponential(80, n_fraud) + 20,
        "std_amount_last_7d": rng.exponential(60, n_fraud),
        "max_amount_last_30d": rng.exponential(500, n_fraud) + 100,
        "time_since_last_transaction": rng.exponential(2, n_fraud),
        "is_weekend": rng.binomial(1, 0.5, n_fraud),
        "is_international": rng.binomial(1, 0.4, n_fraud),
    }

    # --- Combine and shuffle -----------------------------------------------
    df_legit = pd.DataFrame(legit)
    df_legit[TARGET_NAME] = 0

    df_fraud = pd.DataFrame(fraud)
    df_fraud[TARGET_NAME] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Ensure correct column order
    df = df[FEATURE_NAMES + [TARGET_NAME]]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Dataset saved to {save_path} ({len(df):,} rows, {df[TARGET_NAME].sum():,} fraud)")

    return df


def load_data(path: str) -> pd.DataFrame:
    """Load transaction data from CSV."""
    df = pd.read_csv(path)
    expected = set(FEATURE_NAMES + [TARGET_NAME])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df
