# main.py
import argparse
from src.data_loader import load_data, get_data_split
from src.train import train_model
from src.evaluation import evaluate_model, run_shap_analysis, calculate_financial_impact

def main():
    # Argument parser for CLI control
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP explainability plots")
    args = parser.parse_args()

    print("🚀 Starting Churn Prediction Pipeline...")

    # 1. Load Data
    df = load_data()

    # 2. Split Data
    X_train, X_test, y_train, y_test = get_data_split(df)

    # 3. Train (with optional tuning)
    pipeline = train_model(X_train, y_train, X_test, y_test, tune_optuna=args.tune)

    print("🔮 Generating Predictions...")

    # 4. Evaluate and get predictions
    eval_results = evaluate_model(pipeline, X_test, y_test)
    y_pred = eval_results['y_pred']

    # 💰 Calculate financial impact using the predictions
    calculate_financial_impact(y_test, y_pred)

    # 5. Explain (Optional)
    if args.shap:
        run_shap_analysis(pipeline, X_test)

    print("\n✅ Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()