import datetime
import os
import pickle
import random

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "model.pkl"
ML_MODEL = None
ML_FEATURES = []


def load_model():
    """Load trained model if it exists."""
    global ML_MODEL, ML_FEATURES

    if not os.path.exists(MODEL_PATH):
        return

    try:
        with open(MODEL_PATH, "rb") as model_file:
            saved_data = pickle.load(model_file)

        ML_MODEL = saved_data.get("model")
        ML_FEATURES = saved_data.get("features", [])
    except Exception:
        # Keep app running even if model loading fails.
        ML_MODEL = None
        ML_FEATURES = []


def fallback_prediction():
    """Simple fallback when model is not available."""
    traffic_ranges = {
        "low": (3, 6),
        "medium": (6, 10),
        "high": (10, 15),
    }
    traffic_level = random.choice(list(traffic_ranges.keys()))
    min_eta, max_eta = traffic_ranges[traffic_level]
    eta = random.randint(min_eta, max_eta)
    explanation = "ML model unavailable, using fallback estimate."
    return eta, traffic_level, explanation


def default_traffic_by_hour(hour_of_day):
    """Return a simple traffic guess by hour."""
    if hour_of_day in (7, 8, 9, 17, 18, 19):
        return "high"
    if hour_of_day in (6, 10, 16, 20):
        return "medium"
    return "low"


def predict_with_model(bus_line, stop):
    """Predict ETA using trained model and default context values."""
    now = datetime.datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    traffic_level = default_traffic_by_hour(hour_of_day)

    sample_data = {
        "line_id": bus_line or "L01",
        "stop_id": stop or "STOP_1",
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "weather_condition": "clear",
        "traffic_level": traffic_level,
        "cumulative_delay_min": 4.0,
        "speed_factor": 1.0,
    }

    model_input = {feature: sample_data.get(feature) for feature in ML_FEATURES}
    input_df = pd.DataFrame([model_input])

    prediction = ML_MODEL.predict(input_df)[0]
    eta = max(1, int(round(float(prediction))))
    explanation = "Prediction comes from the trained ML model."
    return eta, traffic_level, explanation


load_model()


@app.route("/", methods=["GET", "POST"])
def home():
    eta = None
    traffic_level = None
    explanation = None
    bus_line = ""
    stop = ""

    if request.method == "POST":
        bus_line = request.form.get("bus_line", "").strip()
        stop = request.form.get("stop", "").strip()

        if ML_MODEL is not None and ML_FEATURES:
            try:
                eta, traffic_level, explanation = predict_with_model(bus_line, stop)
            except Exception:
                eta, traffic_level, explanation = fallback_prediction()
        else:
            eta, traffic_level, explanation = fallback_prediction()

    return render_template(
        "index.html",
        eta=eta,
        traffic_level=traffic_level,
        explanation=explanation,
        bus_line=bus_line,
        stop=stop,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
