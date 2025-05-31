import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)
    return df


def split_data(df: pd.DataFrame):
    year = pd.to_datetime(df["Date"]).dt.year
    return (df[year < 2015], df[year == 2015], df[year > 2015])
