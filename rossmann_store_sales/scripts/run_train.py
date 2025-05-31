import sys
import os
import logging

# Add the src directory to the Python pat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_data
from feature_engineering import split_date, comp_months, promo_cols
from train import train_model
from preprocessing import create_preprocessor
from config import (
    DATA_PATH,
    MODEL_PATH,
    MODEL_NAME,
    INPUT_FEATURES,
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from model_registry import MODEL_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("train.log"),  # Also save to log file
    ],
)

if __name__ == "__main__":
    logging.info("Loading datasets...")
    train_path = os.path.join(DATA_PATH, "train.csv")
    store_path = os.path.join(DATA_PATH, "store.csv")
    test_path = os.path.join(DATA_PATH, "test.csv")

    # Load data
    train_df, test_df = load_data(train_path, store_path, test_path)

    # Feature engineering
    for df in [train_df, test_df]:
        split_date(df)
        comp_months(df)
        promo_cols(df)

    # Impute missing CompetitionDistance using max value
    max_distance = train_df["CompetitionDistance"].max()
    train_df.fillna({"CompetitionDistance": max_distance}, inplace=True)
    test_df.fillna({"CompetitionDistance": max_distance}, inplace=True)

    logging.info("Training model...")
    preprocessor = create_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    X = train_df[INPUT_FEATURES]
    target = train_df[TARGET]
    y = test_df[INPUT_FEATURES]

    pipeline = train_model(
        X, target, preprocessor, MODEL_REGISTRY["xgb_regressor"], MODEL_PATH, MODEL_NAME
    )
    logging.info(f"Model saved to {MODEL_PATH}")
