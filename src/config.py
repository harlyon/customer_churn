import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Bank_Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
REPORTS_PATH = BASE_DIR / "reports"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

ID_COLS = ["CustomerId", "Surname"]
TARGET_COL = "Exited"

NUMERICAL_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
CATEGORICAL_FEATURES = ['Geography', 'Gender']
PASSTHROUGH_FEATURES = ['HasCrCard', 'IsActiveMember']

XGB_PARAMS = {
    'n_estimators': 455,
    'learning_rate': 0.12,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}