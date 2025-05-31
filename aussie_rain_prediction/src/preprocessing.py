import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_columns(df):
    input_cols = list(df.columns)[1:-1]
    numeric_cols = df[input_cols].select_dtypes(include=np.number).columns.tolist()[:-1]
    categorical_cols = df[input_cols].select_dtypes("object").columns.tolist()
    return input_cols, numeric_cols, categorical_cols


def create_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
