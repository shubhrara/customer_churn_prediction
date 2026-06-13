import os
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocessing import get_preprocessor, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL

def train_and_tune():
    # Load processed data
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed files not found. Run preprocessing.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Get preprocessor
    preprocessor = get_preprocessor()
    
    # Define models and hyperparameter grids
    # We apply preprocessor to training data first for GridSearchCV speed,
    # then we fit the final pipeline on the original text-like columns.
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
    }
    
    best_estimators = {}
    best_scores = {}
    
    print("Starting hyperparameter tuning...")
    for name, config in models.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="f1",  # F1-score is robust to class imbalance
            n_jobs=-1
        )
        grid.fit(X_train_preprocessed, y_train)
        best_estimators[name] = grid.best_estimator_
        best_scores[name] = grid.best_score_
        print(f"Best parameters for {name}: {grid.best_params_}")
        print(f"Best CV F1-score for {name}: {grid.best_score_:.4f}")
        
    # Select the best model based on CV F1 score
    best_model_name = max(best_scores, key=best_scores.get)
    print(f"\nBest model overall based on CV F1-score: {best_model_name}")
    
    best_clf = best_estimators[best_model_name]
    
    # Save preprocessing component and model combined as a single pipeline
    # We fit the pipeline on the original training data
    churn_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_clf)
    ])
    
    # Refit pipeline on full training data (original columns)
    churn_pipeline.fit(X_train, y_train)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    pipeline_path = os.path.join("models", "churn_pipeline.pkl")
    
    with open(pipeline_path, "wb") as f:
        pickle.dump(churn_pipeline, f)
        
    print(f"Saved completed inference pipeline to: {pipeline_path}")
    
    # Also save training performance summary for downstream use
    summary_path = os.path.join("models", "training_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump({
            "best_model_name": best_model_name,
            "best_scores": best_scores,
            "models": best_estimators,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }, f)
        
    print("Saved training summary for evaluation script.")

if __name__ == "__main__":
    train_and_tune()
