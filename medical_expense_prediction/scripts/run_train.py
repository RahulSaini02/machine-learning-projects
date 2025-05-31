import sys
import os
import logging

# Add the src directory to the Python pat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_data
from train import train_model
from evaluate import evaluate_model

DATA_PATH = "data/medical-charges.csv"
MODEL_PATH = "models/pipeline.joblib"

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
    logging.info("Loading dataset...")
    data = load_data(DATA_PATH)

    logging.info("Training model...")
    model, X_test, y_test = train_model(data, MODEL_PATH)

    logging.info("Evaluating model...")
    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(
        f"Evaluation Results -> MSE: {mse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}"
    )
