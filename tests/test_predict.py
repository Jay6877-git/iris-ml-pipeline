# tests/test_predict.py
import pytest
from sklearn.model_selection import train_test_split

# from your project
from src.iris_model.train import load_data, train_model
from src.iris_model.predict import predict  # no need to import load_model here


@pytest.fixture
def loaded_model():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return train_model(X_tr, y_tr)  # return the trained pipeline directly


def test_predict_happy_path(loaded_model):
    samples = [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 6.0, 2.5]]
    preds = predict(samples, loaded_model)

    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert p in (0, 1, 2)

    one = predict([5.1, 3.5, 1.4, 0.2], loaded_model)
    assert isinstance(one, list)
    assert one[0] in (0, 1, 2)


@pytest.mark.parametrize(
    "bad_samples",
    [
        [[5.1, 3.5, 1.4]],  # wrong length
        [[5.1, 3.5, 1.4, "oops"]],  # non-numeric
    ],
)
def test_predict_invalid_inputs(loaded_model, bad_samples):
    try:
        out = predict(bad_samples, loaded_model)
    except Exception:
        return
    assert out is None
