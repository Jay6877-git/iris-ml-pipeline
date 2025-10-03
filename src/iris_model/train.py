"""
Training script for the Iris Classifier.

This module:
1. Loads the Iris dataset (classic dataset with 150 flower samples).
2. Splits the data into training and test sets.
3. Builds a machine learning pipeline (scaler + logistic regression).
4. Trains the pipeline on training data.
5. Evaluates it on test data (accuracy + classification report).
6. Saves the trained model and evaluation metrics to disk.

Artifacts:
- models/iris.joblib      : serialized trained pipeline
- artifacts/metrics.json  : evaluation results (accuracy + report)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import json


def load_data():
    """Load features (X) and labels (y) from sklearn's Iris dataset."""
    data = load_iris()
    x = data.data
    y = data.target
    return x, y


def train_model(X_train, Y_train):
    """
    Build a pipeline with preprocessing + classifier and fit it.

    Steps:
    - StandardScaler: normalize features (zero mean, unit variance).
    - LogisticRegression: classifier (max_iter=300 ensures convergence).
    """
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=300, random_state=42)

    # Pipeline ties preprocessing and model together
    pipeline = Pipeline([("scale", scaler), ("clf", model)])

    # Train the pipeline
    pipeline.fit(X_train, Y_train)
    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate a trained model on test data.

    Returns:
        metrics (dict): includes accuracy (float) and
                        classification report (dict with precision/recall/f1).
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred, output_dict=True)

    metrics = {"accuracy": accuracy, "report": report}

    return metrics


def save_model(model, model_Path):
    """
    Save the trained model pipeline to disk.

    Args:
        model: trained sklearn pipeline
        model_Path (str): folder name (relative to project root) to store the model
    """
    model_Path.mkdir(parents=True, exist_ok=True)

    model_path = model_Path / "iris.joblib"
    joblib.dump(model, model_path)


def save_artifacts(metrics, artifacts_dir_path):
    """
    Save evaluation metrics as JSON to disk.

    Args:
        metrics (dict): contains accuracy and classification report
        artifacts_dir_name (str): folder name (relative to project root)
    """
    artifacts_dir_path.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir_path / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)


def main():
    """Orchestrates the full training + evaluation + saving pipeline."""
    # 1. Load dataset
    x, y = load_data()

    # 2. Split into train/test sets (80/20 split, stratified for class balance)
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train pipeline
    pipeline = train_model(X_train, Y_train)

    # 4. Evaluate
    metrics = evaluate_model(pipeline, X_test, Y_test)

    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models"
    # 5. Save model + metrics
    save_model(pipeline, model_path)

    artifacts_path = ROOT / "artifacts"
    save_artifacts(metrics, artifacts_path)

    # 6. Print concise summary
    print(f"✅ Training complete. Accuracy: {metrics['accuracy']:.3f}")
    print("📦 Model saved to: models/iris.joblib")
    print("📊 Metrics saved to: artifacts/metrics.json")


if __name__ == "__main__":
    main()
