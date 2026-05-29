import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def evaluate_model(y_true, y_pred, y_prob):
    """Calculate evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def train_and_track(model_name, model, X_train, X_test, y_train, y_test):
    """Train a model and log results to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_prob)
        
        # Log Parameters
        mlflow.log_param("model_type", model_name)
        
        # Log Metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"{model_name} {metric_name}: {value:.4f}")
            
        # Log Model
        mlflow.sklearn.log_model(model, model_name)
        
        return metrics

if __name__ == "__main__":
    # 1. Load Processed Data
    data_path = 'data/processed/processed_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run task 4 first.")
        exit()
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlflow.set_tracking_uri(f"file:///{os.path.join(os.getcwd(), 'mlruns').replace('\\', '/')}")
    # 3. Initialize MLflow Experiment
    mlflow.set_experiment("Credit_Risk_Scoring")
    
    # 4. Train Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    train_and_track("Logistic_Regression", lr_model, X_train, X_test, y_train, y_test)
    
    # 5. Train Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_track("Random_Forest", rf_model, X_train, X_test, y_train, y_test)

    print("\nMLflow Tracking Complete. Run 'mlflow ui' to see results.")