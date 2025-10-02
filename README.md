# Iris Classifier (Scikit-learn + CI/CD)

A small machine learning project that predicts the species of iris flowers based on their measurements. 
Built to practice Git, testing, pre-commit, and GitHub Actions CI.

## 📂 Project Structure
```
iris-ml-pipeline/
│
├── src/
│   └── iris_model/
│       └── train.py       # Training script
│       └── predict.py     # Prediction script
│
├── models/                # Saved trained models (created after training)
│   └── iris.joblib
│
├── artifacts/             # Saved metrics (created after training)
│   └── metrics.json
│
├── .venv/                 # Virtual environment (not committed)
└── README.md
```

## ⚙️ Setup

1. Clone the repo
```bash
git clone https://github.com/your-username/iris-ml-pipeline.git
cd iris-ml-pipeline
```
2. Create virtual environment
```
python -m venv .venv
source .venv/Scripts/activate
```

3. Install dependencies
```
python -m pip install --upgrade pip
python -m pip install scikit-learn numpy joblib pytest
```

## 🚀 Run Training
```
python -m src.iris_model.train
```
## Example output:
```
✅ Training complete. Accuracy: 0.933
📦 Model saved to: models/iris.joblib
📊 Metrics saved to: artifacts/metrics.json
```

## 🔮 Run Prediction
Once training is done and models/iris.joblib exists, you can run predictions.

### Predict a single sample from CLI
```
python -m src.iris_model.predict 5.1 3.5 1.4 0.2
```

Example output:
```
Predicted class: 0
```
➡️ Class labels are integers (0, 1, 2), corresponding to the Iris dataset species.

### Predict multiple samples (in code)
```
from src.iris_model.predict import load_model, predict

model = load_model()
samples = [
    [5.1, 3.5, 1.4, 0.2],   # Iris-setosa-like
    [6.7, 3.0, 5.2, 2.3]    # Iris-virginica-like
]
print(predict(samples, model))
# Output: [0, 2]
```