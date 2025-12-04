import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score
from src.config import REPORTS_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, PASSTHROUGH_FEATURES, AVERAGE_CLV, CAMPAIGN_COST

def calculate_financial_impact(y_test, y_pred):
    """
    Calculates the estimated ROI of using the model vs doing nothing.
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 1. Costs
    # We spend money on everyone we predict as churn (TP + FP)
    total_intervention_cost = (tp + fp) * CAMPAIGN_COST

    # 2. Revenue Saved
    # We save the CLV of the churners we correctly caught (TP)
    # Note: This assumes 100% success rate of the intervention.
    # In reality, maybe only 50% stay even after a discount.
    revenue_saved = tp * AVERAGE_CLV

    # 3. Net Profit
    net_profit = revenue_saved - total_intervention_cost

    # 4. ROI %
    roi = (net_profit / total_intervention_cost) * 100 if total_intervention_cost > 0 else 0

    print("\n--- 💰 Financial Impact Report ---")
    print(f"Assumption: CLV=${AVERAGE_CLV}, Cost=${CAMPAIGN_COST}")
    print(f"-----------------------------------")
    print(f"🎯 Targeted Customers: {tp + fp} (TP + FP)")
    print(f"💸 Total Campaign Cost: ${total_intervention_cost:,.2f}")
    print(f"🛡️ Revenue Saved:       ${revenue_saved:,.2f}")
    print(f"-----------------------------------")
    print(f"📈 NET PROFIT:          ${net_profit:,.2f}")
    print(f"🚀 ROI:                 {roi:.1f}%")
    print(f"-----------------------------------\n")

    return net_profit

def evaluate_model(pipeline, X_test, y_test):
    """Generates metrics, saves plots, and prints report.

    Returns:
        dict: Dictionary containing 'roc_auc', 'recall', and 'y_pred'
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Console Report
    print("\n--- 📊 Final Evaluation ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall (Churners): {recall:.4f}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(REPORTS_PATH / "confusion_matrix.png")
    plt.close()

    # 3. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(REPORTS_PATH / "roc_curve.png")
    plt.close()

    print(f"📉 Plots saved to {REPORTS_PATH}")

    # Return metrics and predictions
    return {
        'roc_auc': auc,
        'recall': recall,
        'y_pred': y_pred
    }

def run_shap_analysis(pipeline, X_test):
    """Generates SHAP summary plot."""
    print("🧠 Generating SHAP explanations...")

    # 1. Transform data
    # We must explicitly transform X_test because SHAP works on the model input, not raw data
    preprocessor = pipeline.named_steps['preprocessor']
    X_test_trans = preprocessor.transform(X_test)

    # 2. Reconstruct Feature Names
    # Note: This relies on the specific order in preprocessing.py
    feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES

    # 3. Explain
    model = pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_trans)

    # 4. Save Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
    plt.savefig(REPORTS_PATH / "shap_summary.png", bbox_inches='tight')
    plt.close()
    print("🧠 SHAP plot saved.")