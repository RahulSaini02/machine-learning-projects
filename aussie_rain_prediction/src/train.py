from sklearn.pipeline import Pipeline
import joblib
import os


def train_model(X, y, preprocessor, classifier, model_path, model_name):
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X, y)

    file_path = os.path.join(model_path, model_name)
    joblib.dump(pipeline, file_path)
    return pipeline
