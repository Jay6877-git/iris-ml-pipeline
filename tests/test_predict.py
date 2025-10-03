# tests/test_predict.py
import pytest

# from your project
from src.iris_model.predict import load_model, predict


def test_predict_happy_path():
    # Arrange: ensure a model exists and load it
    model = load_model()
    assert model is not None, "Expected a trained model to be loadable"

    # Act: two valid samples (shape n×4, numeric)
    samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 3.3, 6.0, 2.5],
    ]
    preds = predict(samples, model)

    # Assert
    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert p in (0, 1, 2)


@pytest.mark.parametrize(
    "bad_samples",
    [
        [[5.1, 3.5, 1.4]],  # wrong length (only 3 features)
        [[5.1, 3.5, 1.4, "oops"]],  # non-numeric
    ],
)
def test_predict_invalid_inputs(bad_samples):
    # Arrange: ensure a model exists and load it
    model = load_model()
    assert model is not None, "Expected a trained model to be loadable"

    # Act + Assert:
    # Your predict() may either raise (ValueError/Exception)
    # OR return None with a printed message. Accept either behavior.
    try:
        out = predict(bad_samples, model)
    except Exception:
        return  # raising is acceptable behavior

    # If no exception, assert it signaled failure via None
    assert out is None, "Expected predict() to fail (None) for invalid input"
