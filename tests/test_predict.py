# tests/test_predict.py
import pytest
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# from your project
from src.iris_model.train import load_data, train_model
import src.iris_model.predict as P
from src.iris_model.predict import predict


@pytest.fixture
def loaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Train a small pipeline, save it to tmp_path/models/iris.joblib,
    and patch predict.py to load from that temp root.
    Returns a loaded, usable model.
    """
    # Train a tiny model
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = train_model(X_train, y_train)

    # Save to the temp "project root"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Write the exact filename load_model() expects
    joblib.dump(pipeline, models_dir / "iris.joblib")

    # Patch predict.py's root so load_model() looks under tmp_path
    monkeypatch.setattr(P, "ROOT", tmp_path, raising=False)

    # Load via the actual function under test (uses patched ROOT)
    model = P.load_model()
    assert model is not None, "Expected a trained model to be loadable from tmp_path"
    return model


def test_predict_happy_path(loaded_model):
    # 2D, two valid samples
    samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 3.3, 6.0, 2.5],
    ]
    preds = predict(samples, loaded_model)

    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert p in (0, 1, 2)

    # (If your predict() supports 1D, you can also check a single sample:)
    one = predict([5.1, 3.5, 1.4, 0.2], loaded_model)
    assert isinstance(one, list) and one[0] in (0, 1, 2)


@pytest.mark.parametrize(
    "bad_samples",
    [
        [[5.1, 3.5, 1.4]],  # wrong length (3 features)
        [[5.1, 3.5, 1.4, "oops"]],  # non-numeric
    ],
)
def test_predict_invalid_inputs(loaded_model, bad_samples):
    # Your predict() may raise OR return None for invalid input; accept either.
    try:
        out = predict(bad_samples, loaded_model)
    except Exception:
        return
    assert out is None, "Expected predict() to fail (None) for invalid input"
