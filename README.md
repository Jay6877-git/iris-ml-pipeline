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
✅ Training complete. Accuracy: 0.933
📦 Model saved to: models/iris.joblib
📊 Metrics saved to: artifacts/metrics.json