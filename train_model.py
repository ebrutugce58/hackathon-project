from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "stop_arrivals.csv"
MODEL_PATH = "model.pkl"
TARGET_COLUMN = "delay_min"

CANDIDATE_FEATURES = [
    "line_id",
    "stop_id",
    "hour_of_day",
    "day_of_week",
    "weather_condition",
    "traffic_level",
    "speed_factor",
]


def main():
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        print(f"File not found: {DATA_PATH}")
        print("Please add stop_arrivals.csv and run again.")
        return

    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    if TARGET_COLUMN not in df.columns:
        print("Target column 'delay_min' was not found.")
        return

    feature_columns = [col for col in CANDIDATE_FEATURES if col in df.columns]
    if not feature_columns:
        print("No matching feature columns were found in the dataset.")
        return

    print(f"Using features: {feature_columns}")
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN]

    # Fill empty values in categorical columns so encoder gets clean strings.
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("unknown")

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    print(f"Categorical features: {categorical_features}")
    print(f"Numeric features: {numeric_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Training complete. MAE: {mae:.2f}")

    joblib.dump({"model": model, "features": feature_columns}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
