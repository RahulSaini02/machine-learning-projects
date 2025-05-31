import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import load_data, split_data
from preprocessing import get_feature_columns, create_preprocessor
from train import train_model
from config import TARGET_COL
from model_registry import MODEL_REGISTRY

DATA_PATH = "data/weatherAUS.csv"
MODEL_PATH = "models"
EXPERIMENT_NAME = "aussie_rain_prediction"
MODEL_REGISTRY_NAME = "AussieRainModel"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    logging.info("Loading data...")
    raw_df = load_data(DATA_PATH)
    train_df, val_df, test_df = split_data(raw_df)

    input_cols, numeric_cols, categorical_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[input_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[input_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[input_cols], test_df[TARGET_COL]

    preprocessor = create_preprocessor(numeric_cols, categorical_cols)

    # Experiment tracking
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Train and evaluate each model
    for name, clf in MODEL_REGISTRY.items():
        with mlflow.start_run(run_name=name):
            logging.info(f"Training model: {name}")

            pipeline = train_model(
                X_train, y_train, preprocessor, clf, MODEL_PATH, f"{name}.joblib"
            )

            # example data
            input_example = X_val.head(5)  # raw input example
            output_example = pipeline.predict(input_example)
            signature = infer_signature(input_example, output_example)

            # Evaluate
            val_preds = pipeline.predict(X_val)
            test_preds = pipeline.predict(X_test)

            acc_val = accuracy_score(y_val, val_preds)
            acc_test = accuracy_score(y_test, test_preds)

            # Log metrics
            mlflow.log_param("model_name", name)
            mlflow.log_metric("val_accuracy", acc_val)
            mlflow.log_metric("test_accuracy", acc_test)

            # Generate and save confusion matrix
            cm = confusion_matrix(y_val, val_preds, normalize="true")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)
            ax.set_title(f"{name} Confusion Matrix - Validation")
            artifact_dir = tempfile.mkdtemp()
            plot_path = os.path.join(artifact_dir, f"{name}_confusion_matrix.png")
            fig.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")

            # Log and register model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"{MODEL_REGISTRY_NAME}_{name}",
                signature=signature,
                input_example=input_example,
            )
            logging.info(f"{name} logged and registered with MLflow.")

            logging.info(
                f"MLflow logged: {name} | Val Acc: {acc_val:.2f} | Test Acc: {acc_test:.2f}"
            )
