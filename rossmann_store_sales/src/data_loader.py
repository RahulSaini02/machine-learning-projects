# src/data_loader.py

import pandas as pd


def load_data(train_path: str, store_path: str, test_path: str):
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df = train_df.merge(store_df, how="left", on="Store")
    test_df = test_df.merge(store_df, how="left", on="Store")

    # Filter out closed stores
    train_df = train_df[train_df["Open"] == 1].copy()

    return train_df, test_df
