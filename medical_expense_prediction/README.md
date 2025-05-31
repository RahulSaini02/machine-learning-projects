# ☔ Aussie Rain Prediction

This project predicts whether it will rain tomorrow at various locations in Australia using weather history data and classification models.

## 📦 Project Structure

```
aussie_rain_prediction/
│
├── data/
│   └── weatherAUS.csv
│
├── models/
│   └── aussie_rain_prediction.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│
├── scripts/
│   └── run_train.py
│
├── tests/
│   └── test_train.py
│
├── requirements.txt
├── README.md
└── .gitignore

```

## ⚙️ Features

- Feature engineering from date fields
- Separate preprocessing pipelines for numerical and categorical data
- Supports Logistic Regression, Random Forest, and Decision Tree classifiers
- Integrated with MLflow for tracking and model registry
- Confusion matrix visualizations

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python scripts/run_train.py
