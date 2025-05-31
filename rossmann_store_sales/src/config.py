INPUT_FEATURES = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpen",
    "Day",
    "Month",
    "Year",
    "WeekOfYear",
    "Promo2",
    "Promo2Open",
    "IsPromo2Month",
]
TARGET = "Sales"
NUMERIC_FEATURES = [
    "Store",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpen",
    "Promo2",
    "Promo2Open",
    "IsPromo2Month",
    "Day",
    "Month",
    "Year",
    "WeekOfYear",
]
CATEGORICAL_FEATURES = ["DayOfWeek", "StateHoliday", "StoreType", "Assortment"]


DATA_PATH = "data/rossmann-store-sales"
MODEL_PATH = "models"
MODEL_NAME = "rossmann_rf_model.joblib"
