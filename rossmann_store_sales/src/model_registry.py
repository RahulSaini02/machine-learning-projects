from xgboost import XGBRegressor

MODEL_REGISTRY = {
    "xgb_regressor": XGBRegressor(
        n_jobs=-1,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.2,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.7,
    ),
}
