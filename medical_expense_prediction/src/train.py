import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from config import TARGET, RANDOM_STATE
from preprocessing import create_preprocessor


def train_model(data: pd.DataFrame, model_path: str):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = create_preprocessor()
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    return model, X_test, y_test
