import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "stop_arrivals.csv"
MODEL_PATH = "model.pkl"

# Preferred features (use only those available in the CSV).
PREFERRED_FEATURES = [
    "line_id",
    "stop_id",
    "hour_of_day",
    "day_of_week",
    "weather_condition",
    "traffic_level",
    "cumulative_delay_min",
    "speed_factor",
]
TARGET_COLUMN = "delay_min"


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Dataset must contain 'delay_min' column.")

    feature_columns = [feature for feature in PREFERRED_FEATURES if feature in df.columns]
    if not feature_columns:
        raise ValueError("None of the preferred features were found in the dataset.")

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    categorical_features = [col for col in feature_columns if X[col].dtype == "object"]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained successfully. MAE: {mae:.2f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump({"model": model, "features": feature_columns}, model_file)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
