# scripts/train.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the experiment name
#mlflow.set_tracking_uri("sqlite:///mlflow.db")
#mlflow.set_experiment("Iris_Classification")

# --- NEW SELF-CONTAINED SETUP ---
# Set the tracking URI to the SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set the experiment. If it doesn't exist, MLflow creates it
# AND sets its artifact location to the correct relative path.
# mlflow.set_experiment(experiment_name="Iris_Classification", artifact_location="mlruns")
# Use a simple relative path)

mlflow.set_experiment("Iris_Classification")

# --- END NEW SETUP ---
df = pd.read_csv("data/processed/iris_processed.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Logistic Regression ---
with mlflow.start_run(run_name="LogisticRegression"):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(lr, "model")
    print(f"Logistic Regression Accuracy: {accuracy}")

# --- Model 2: Random Forest ---
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(rf, "model")
    print(f"Random Forest Accuracy: {accuracy}")