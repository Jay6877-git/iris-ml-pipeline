"""
Prediction script for the Iris Classifier.

This module:
- Loads the trained pipeline model from disk.
- Provides a `predict()` function to run predictions on new samples.
- Validates input shape and type (1D single sample or 2D multiple samples).
- Can be run directly as a CLI with 4 numeric arguments.

Artifacts expected:
- models/iris.joblib : serialized trained pipeline (created in training phase).
"""

from pathlib import Path
import sys
import numpy as np
import joblib


def load_model():
    """
    Load the trained iris model from the models/iris.joblib file.

    Returns:
        Trained sklearn pipeline, or None if the file is missing.
    """
    ROOT = Path(__file__).resolve().parents[2]
    model_path = ROOT / "models/iris.joblib"
    if not model_path.exists():
        print("Model file not found, please run training first.")
        return None
    else:
        return joblib.load(model_path)

def predict(samples, model):
    """
    Predict the class of new iris samples using the trained model.

    Args:
        samples (list or array-like): Either
            - Single sample: [sepal_len, sepal_wid, petal_len, petal_wid]
            - Multiple samples: [[...], [...], ...]
        model: Trained sklearn pipeline (with scaler + classifier).

    Returns:
        List of predicted class IDs, or None if validation fails.
    """
    numpy_sample = np.array(samples)

    # Case 1: single sample passed as 1D list
    if numpy_sample.ndim == 1:
        if numpy_sample.shape[0] == 4:
            # reshape to (1, 4) so sklearn accepts it
            numpy_sample = numpy_sample.reshape(1, -1)
        else:
            print("Invalid input: expected 4 numbers for one sample.")
            return None
    
    # Case 2: multiple samples passed as 2D array
    elif numpy_sample.ndim == 2:
        n, m = numpy_sample.shape

        if m != 4:
            print("Invalid input: each sample must have exactly 4 numbers (sepal_length, sepal_width, petal_length, petal_width).")
            return None
    
        try:
            numpy_sample = numpy_sample.astype(float)
        except ValueError:
            print("Invalid input: all values must be numeric (got 'abc' instead).")
            return None
    
    # Case 3: higher-dimensional input → reject
    else:
        print("Invalid input: expected 1D (single) or 2D (multiple) samples.")
        return None
    
    # Make predictions with the model
    predictions =  model.predict(numpy_sample)

    return predictions.tolist()

if __name__ == "__main__":
    """
    CLI entrypoint.

    Usage:
        python -m src.iris_model.predict <sepal_len> <sepal_wid> <petal_len> <petal_wid>
    """
    # this block only runs if file is executed directly
    args = sys.argv[1:]   # collect command-line arguments (skip script name)

    if len(args) == 4:
        # Validate that all 4 inputs are numeric
        try:
            sample = [float(a) for a in args]
        except ValueError:
            print("Invalid input: all values must be numeric.")
            sys.exit(1)

        # Load the model
        model = load_model()
        if model is None:
            sys.exit(1)

        # Run prediction
        preds = predict(sample, model)
        if preds is None:
            sys.exit(1)
        else:
            print(f"Predicted class: {preds[0]}")
    
    else:
        # Wrong number of args → show usage
        print("Usage: python -m src.iris_model.predict <sepal_len> <sepal_wid> <petal_len> <petal_wid>")
