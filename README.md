# 🔍 Fraud Detection with Explainable AI (XAI)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning pipeline for **detecting fraudulent financial transactions** with built-in **SHAP-based explainability**. Unlike black-box fraud detectors, this system tells you *why* each transaction was flagged — making it audit-ready and trustworthy for production deployment.

<p align="center">
  <img src="docs/demo.gif" alt="Demo" width="800">
</p>

## 🧠 Why Explainable AI Matters in Fraud Detection

Traditional ML models flag transactions as "fraud" or "not fraud" — but **regulators, compliance teams, and customers need to know why**. This project integrates [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) to provide:

- **Global explanations** — Which features drive fraud detection overall?
- **Local explanations** — Why was *this specific transaction* flagged?
- **Audit trail** — Every prediction comes with a traceable feature contribution breakdown

## 🏗️ Architecture

```
┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────────┐
│  Synthetic   │───▶│ Preprocessing  │───▶│   XGBoost    │───▶│     SHAP       │
│  Data Gen    │    │  + SMOTE       │    │   Classifier │    │  Explainability │
└──────────────┘    └────────────────┘    └──────────────┘    └────────────────┘
       │                    │                     │                    │
       ▼                    ▼                     ▼                    ▼
  50K realistic       StandardScaler         Early stopping      Beeswarm plots
  transactions        Stratified split       Class weighting     Waterfall plots
  2% fraud ratio      SMOTE oversampling     PR-AUC optimized   Feature importance
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Realistic Data Generation** | 50K synthetic transactions with 20 engineered features simulating real fraud patterns |
| **Class Imbalance Handling** | SMOTE/ADASYN oversampling with stratified splits |
| **XGBoost Classifier** | Gradient-boosted trees with early stopping and PR-AUC optimization |
| **SHAP Explainability** | TreeExplainer for global & local feature attributions |
| **Interactive Demo** | Streamlit app with real-time scoring and visual explanations |
| **Configurable Pipeline** | YAML-driven config for reproducible experiments |
| **Docker Support** | One-command deployment with Dockerfile |
| **Test Suite** | pytest-based unit tests covering the full pipeline |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/tokumag/fraud-detection-xai.git
cd fraud-detection-xai

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

This will:
1. Generate 50K synthetic transactions (2% fraud rate)
2. Preprocess with StandardScaler + SMOTE oversampling
3. Train an XGBoost classifier with early stopping
4. Evaluate with precision, recall, F1, ROC-AUC, PR-AUC
5. Generate SHAP explainability plots
6. Save all artifacts to `models/`

### Launch the Demo App

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t fraud-detection-xai .
docker run -p 8501:8501 fraud-detection-xai
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.99 |
| PR-AUC | ~0.98 |
| Precision | ~0.95 |
| Recall | ~0.96 |
| F1 Score | ~0.95 |

> **Note:** Scores are on synthetic data and serve to validate the pipeline.
> Real-world performance would depend on the production dataset.

## 📂 Project Structure

```
fraud-detection-xai/
├── app.py                  # Streamlit interactive demo
├── train.py                # Training pipeline entry point
├── config.yaml             # Experiment configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Synthetic data generation
│   ├── preprocessor.py     # Scaling + SMOTE pipeline
│   ├── model.py            # Model training & evaluation
│   └── explainer.py        # SHAP explainability engine
├── tests/
│   └── test_pipeline.py    # Unit test suite
├── data/                   # Generated datasets (git-ignored)
└── models/                 # Trained model artifacts (git-ignored)
```

## 🔧 Configuration

All hyperparameters are controlled via `config.yaml`:

```yaml
model:
  algorithm: xgboost         # xgboost | random_forest | logistic_regression
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: auto   # Auto-calculated from class ratio
    eval_metric: aucpr
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 🛠️ Tech Stack

- **ML Framework:** scikit-learn, XGBoost
- **Explainability:** SHAP, LIME
- **Imbalanced Learning:** imbalanced-learn (SMOTE, ADASYN)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **App Framework:** Streamlit
- **Containerization:** Docker

## 📈 Future Enhancements

- [ ] Add real-world dataset support (Kaggle Credit Card Fraud)
- [ ] Implement LIME explanations alongside SHAP
- [ ] Add model comparison dashboard (XGBoost vs. RF vs. LR)
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Add API endpoint with FastAPI
- [ ] Integrate MLflow for experiment tracking

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tokuma Gebissa** — AI Engineering Student

- GitHub: [@tokumag](https://github.com/tokumag)
- LinkedIn: [Connect with me](https://linkedin.com/in/)

---

*Built with ❤️ as part of an AI Engineering portfolio. If you found this useful, consider giving it a ⭐!*
