from pathlib import Path
import json
import pytest

from src.iris_model.train import (
    load_data,
    train_model,
    evaluate_model,
    save_model,
    save_artifacts,
)

from sklearn.model_selection import train_test_split


def test_training_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Runs the train/eval/save flow in an isolated temp directory and checks:
    - model and metrics files are written
    - metrics dict looks sane
    """
    # Make the temp directory act like the project root for save_model/save_artifacts
    monkeypatch.chdir(tmp_path)

    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    pipeline = train_model(X_train, Y_train)

    metrics = evaluate_model(pipeline, X_test, Y_test)

    # INSERT THESE TWO LINES
    save_model(pipeline, tmp_path / "models")
    save_artifacts(metrics, tmp_path / "artifacts")

    # Assert: files exist in the repo root (because save_* uses parents[2])
    model_path = tmp_path / "models" / "iris.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics.json"

    assert model_path.exists(), "Expected models/iris.joblib to be created"
    assert metrics_path.exists(), "Expected artifacts/metrics.json to be created"

    # Check metrics sanity
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics and "report" in metrics

    acc = metrics["accuracy"]
    report = metrics["report"]

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.85

    assert isinstance(report, dict)

    # Cross-check the saved JSON has the same shape
    saved = json.loads(metrics_path.read_text())
    assert "accuracy" in saved and "report" in saved
    assert isinstance(saved["accuracy"], (int, float))
    assert isinstance(saved["report"], dict)

    # Cleanup (so tests don't pollute your repo)
    try:
        model_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
        (tmp_path / "models").rmdir()
        (tmp_path / "artifacts").rmdir()
    except Exception:
        # if dirs aren't empty or something else wrote there, skip cleanup
        pass
