import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.config import XGB_PARAMS, MODEL_PATH
from src.preprocessing import create_preprocessor

def train_model(X_train, y_train, X_test=None, y_test=None, tune_optuna=False):
    """
    Builds pipeline, optionally tunes hyperparameters, trains, and saves model.
    """

    # Calculate scale_pos_weight dynamically
    # Logic: (Count of 0s) / (Count of 1s)
    neg_count, pos_count = y_train.value_counts().sort_index().values
    scale_pos_weight = neg_count / pos_count
    print(f"⚖️  Class Imbalance Weight: {scale_pos_weight:.2f}")

    preprocessor = create_preprocessor()

    # Update params with calculated weight
    params = XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    if tune_optuna and X_test is not None and y_test is not None:
        print("🔍 Starting Optuna Optimization...")

        def objective(trial):
            # Define search space
            opt_params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1
            }

            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(**opt_params))
            ])
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        print(f"🎯 Best Optuna Params: {study.best_params}")
        params.update(study.best_params)

    # Final Training
    print("🚀 Training Final Pipeline...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))
    ])

    pipeline.fit(X_train, y_train)

    # Save Model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

    return pipeline