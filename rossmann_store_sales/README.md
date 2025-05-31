# 🏪 Rossmann Sales Prediction

This project aims to predict daily sales for Rossmann stores using historical sales data, promotional campaigns, and store information.

## 📦 Project Structure

```
rossmann_sales_prediction/
│
├── data/                         # Raw & processed datasets
│   └── train.csv
│
├── models/                       # Trained models
│   └── rf_model.joblib
│
├── notebooks/                    # EDA & experimentation
│   └── eda.ipynb
│
├── scripts/
│   └── run_train.py              # CLI entry point
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── model_registry.py         
│
├── tests/
│   └── test_train.py
│
├── requirements.txt
├── README.md
└── .gitignore

```


## ⚙️ Features

- Data preprocessing including date and promo feature extraction
- Feature engineering for competition & promotional context
- Trained with Random Forest Regressor
- Modular structure with support for additional models

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python scripts/run_train.py

