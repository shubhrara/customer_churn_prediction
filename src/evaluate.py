import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

def evaluate_models():
    # Load training summary
    summary_path = os.path.join("models", "training_summary.pkl")
    pipeline_path = os.path.join("models", "churn_pipeline.pkl")
    
    if not os.path.exists(summary_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError("Training summary or pipeline not found. Run train.py first.")
        
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
        
    with open(pipeline_path, "rb") as f:
        churn_pipeline = pickle.load(f)
        
    X_train = summary["X_train"]
    y_train = summary["y_train"]
    X_test = summary["X_test"]
    y_test = summary["y_test"]
    trained_models = summary["models"]
    best_model_name = summary["best_model_name"]
    
    preprocessor = churn_pipeline.named_steps["preprocessor"]
    
    # Preprocess test set for custom evaluation of internal models
    X_test_preprocessed = preprocessor.transform(X_test)
    
    metrics_list = []
    
    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for name, clf in trained_models.items():
        # Predict
        y_pred = clf.predict(X_test_preprocessed)
        y_prob = clf.predict_proba(X_test_preprocessed)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    # Finalize ROC Curve Plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    roc_fig_path = os.path.join("reports", "figures", "roc_curve_comparison.png")
    plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC Curve comparison saved to {roc_fig_path}")
    
    # Save comparison table
    metrics_df = pd.DataFrame(metrics_list)
    print("\nModel Evaluation Summary on Test Set:")
    print(metrics_df.to_string(index=False))
    
    metrics_csv_path = os.path.join("reports", "model_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Generate Confusion Matrix for the best model pipeline on test data
    best_y_pred = churn_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {best_model_name} (Best Model)")
    cm_fig_path = os.path.join("reports", "figures", "confusion_matrix.png")
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix saved to {cm_fig_path}")
    
    # Feature Importance for the best model
    # Get feature names from column transformer
    feature_names = preprocessor.get_feature_names_out()
    # Remove prefix like 'num__' and 'cat__' for cleaner display
    clean_feature_names = [f.split("__")[1] if "__" in f else f for f in feature_names]
    
    clf_obj = churn_pipeline.named_steps["classifier"]
    
    importance = None
    importance_type = "Coefficient"
    
    if hasattr(clf_obj, "feature_importances_"):
        importance = clf_obj.feature_importances_
        importance_type = "Importance Score"
    elif hasattr(clf_obj, "coef_"):
        importance = np.abs(clf_obj.coef_[0])
        importance_type = "Absolute Coefficient"
        
    if importance is not None:
        importance_df = pd.DataFrame({
            "Feature": clean_feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
        plt.title(f"Top 15 Features by {importance_type} ({best_model_name})")
        plt.xlabel(importance_type)
        plt.ylabel("Feature")
        plt.grid(True, linestyle="--", alpha=0.3)
        feat_fig_path = os.path.join("reports", "figures", "feature_importance.png")
        plt.savefig(feat_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feature Importance saved to {feat_fig_path}")
        
        # Save feature importance CSV
        importance_df.to_csv(os.path.join("reports", "feature_importance.csv"), index=False)
        
    # Copy figures to Streamlit directory so it can display them easily
    os.makedirs(os.path.join("app", "static"), exist_ok=True)
    import shutil
    for fig_file in ["roc_curve_comparison.png", "confusion_matrix.png", "feature_importance.png"]:
        src_fig = os.path.join("reports", "figures", fig_file)
        if os.path.exists(src_fig):
            shutil.copy(src_fig, os.path.join("app", "static", fig_file))
            
    print("All evaluation plots generated and copied to Streamlit app static directory.")

if __name__ == "__main__":
    evaluate_models()
