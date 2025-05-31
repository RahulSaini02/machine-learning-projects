import pandas as pd


# 1. Split date into components
def split_date(df: pd.DataFrame) -> None:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.isocalendar().week


# 2. CompetitionOpen in number of months
def comp_months(df: pd.DataFrame) -> None:
    df["CompetitionOpen"] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (
        df.Month - df.CompetitionOpenSinceMonth
    )
    df["CompetitionOpen"] = df["CompetitionOpen"].map(lambda x: max(x, 0)).fillna(0)


# 3. Helper to check if promo is active in the given month
def check_promo_month(row) -> int:
    month2str = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sept",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    try:
        months = (row["PromoInterval"] or "").split(",")
        return int(row["Promo2Open"] and month2str[row["Month"]] in months)
    except Exception:
        return 0


# 4. Promotion columns
def promo_cols(df: pd.DataFrame) -> None:
    df["Promo2Open"] = (
        12 * (df.Year - df.Promo2SinceYear)
        + (df.WeekOfYear - df.Promo2SinceWeek) * 7 / 30.5
    )
    df["Promo2Open"] = (
        df["Promo2Open"].map(lambda x: max(x, 0)).fillna(0) * df["Promo2"]
    )
    df["IsPromo2Month"] = df.apply(check_promo_month, axis=1) * df["Promo2"]
