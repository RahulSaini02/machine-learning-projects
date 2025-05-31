import sys
import os
import logging

# Add the src directory to the Python pat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train import train_model
from evaluate import evaluate_model

MODEL_PATH = "models/test_model.joblib"
DATA_PATH = "data/medical-charges.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tests/test_log.log")],
)


@pytest.fixture
def sample_data():
    logging.info("Loading sample data for testing...")
    df = pd.read_csv(DATA_PATH)
    return df


def test_model_training_and_saving(sample_data):
    logging.info("Testing model training and saving...")
    model, X_test, y_test = train_model(sample_data, MODEL_PATH)

    assert isinstance(model, Pipeline), "Model is not a pipeline"
    assert os.path.exists(MODEL_PATH), "Model file was not saved"
    assert X_test.shape[0] > 0, "X_test is empty"
    assert y_test.shape[0] > 0, "y_test is empty"

    logging.info("Model training and saving test passed.")


def test_model_evaluation(sample_data):
    logging.info("Testing model evaluation...")
    model, X_test, y_test = train_model(sample_data, MODEL_PATH)
    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    assert mse >= 0, "MSE cannot be negative"
    assert mae >= 0, "MAE cannot be negative"
    assert -1 <= r2 <= 1, "R² is out of expected range"

    logging.info(
        f"Model evaluation passed. MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}"
    )


def teardown_module(module):
    # Clean up saved model after test
    logging.info("Cleaning up test model file...")
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        logging.info("Model file deleted.")
