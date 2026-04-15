import datetime
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "model.pkl"
BUS_STOPS_PATH = "bus_stops.csv"
ML_MODEL = None
ML_FEATURES = []
MODEL_ERROR = None
BUS_STOPS_DF = None


def load_model():
    """Load trained model if it exists."""
    global ML_MODEL, ML_FEATURES, MODEL_ERROR

    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = "model.pkl not found. Please train the model first."
        return

    try:
        saved_data = joblib.load(MODEL_PATH)
        if isinstance(saved_data, dict):
            ML_MODEL = saved_data.get("model")
            ML_FEATURES = saved_data.get("features", [])
        else:
            ML_MODEL = saved_data
            ML_FEATURES = []
        MODEL_ERROR = None
    except Exception:
        # Keep app running even if model loading fails.
        ML_MODEL = None
        ML_FEATURES = []
        MODEL_ERROR = "Could not load model.pkl. Please retrain the model."


def load_bus_stops():
    """Load bus stop data once at startup."""
    global BUS_STOPS_DF

    if not os.path.exists(BUS_STOPS_PATH):
        BUS_STOPS_DF = None
        return

    try:
        df = pd.read_csv(BUS_STOPS_PATH)
        required_columns = {"line_id", "stop_id"}
        if required_columns.issubset(df.columns):
            BUS_STOPS_DF = df
        else:
            BUS_STOPS_DF = None
    except Exception:
        BUS_STOPS_DF = None


def validate_and_resolve_stop(bus_line, stop_input):
    """
    Validate stop for a bus line.
    - Exact stop_id match is preferred.
    - Partial text can match stop_id within the selected line.
    """
    if BUS_STOPS_DF is None:
        # If stops file is missing, skip strict validation gracefully.
        return True, stop_input, None

    line = (bus_line or "").strip().upper()
    stop_text = (stop_input or "").strip()
    if not line or not stop_text:
        return False, None, "Stop does not belong to this bus line"

    line_stops = BUS_STOPS_DF[BUS_STOPS_DF["line_id"].astype(str).str.upper() == line]
    if line_stops.empty:
        return False, None, "Stop does not belong to this bus line"

    stop_series = line_stops["stop_id"].astype(str)

    # Exact stop_id match.
    exact_matches = line_stops[stop_series.str.upper() == stop_text.upper()]
    if not exact_matches.empty:
        return True, exact_matches.iloc[0]["stop_id"], None

    # Partial text match on stop_id.
    partial_matches = line_stops[stop_series.str.upper().str.contains(stop_text.upper(), na=False)]
    if not partial_matches.empty:
        return True, partial_matches.iloc[0]["stop_id"], None

    return False, None, "Stop does not belong to this bus line"


def build_line_stop_options():
    """Build dropdown data for lines and stops."""
    if BUS_STOPS_DF is None:
        return [], {}

    line_stop_map = {}
    lines = sorted(BUS_STOPS_DF["line_id"].astype(str).str.upper().unique().tolist())

    for line in lines:
        line_rows = BUS_STOPS_DF[
            BUS_STOPS_DF["line_id"].astype(str).str.upper() == line
        ].copy()

        if "stop_sequence" in line_rows.columns:
            line_rows["stop_sequence"] = pd.to_numeric(
                line_rows["stop_sequence"], errors="coerce"
            )
            line_rows = line_rows.sort_values("stop_sequence")

        stops = []
        for _, row in line_rows.iterrows():
            stop_id = str(row.get("stop_id", "")).strip()
            if not stop_id:
                continue

            sequence = row.get("stop_sequence")
            line_name = str(row.get("line_name", "")).strip()
            stop_type = str(row.get("stop_type", "")).strip()

            if pd.notna(sequence):
                stop_text = f"Stop {int(sequence)}"
            else:
                stop_text = "Stop"

            # Friendly label fallback order:
            # 1) Stop {sequence} • {line_name} • {stop_type}
            # 2) Stop {sequence} • {line_name}
            # 3) Stop {sequence}
            if line_name and stop_type:
                label = f"{stop_text} • {line_name} • {stop_type}"
            elif line_name:
                label = f"{stop_text} • {line_name}"
            else:
                label = stop_text

            stops.append({"stop_id": stop_id, "label": label})

        line_stop_map[line] = stops

    return lines, line_stop_map


def predict_with_model(bus_line, stop):
    """Predict ETA using the trained ML model."""
    if ML_MODEL is None:
        return "N/A", "moderate", MODEL_ERROR or "Model is not available.", False

    is_valid, resolved_stop_id, validation_error = validate_and_resolve_stop(bus_line, stop)
    if not is_valid:
        return "Invalid", "moderate", validation_error, False

    now = datetime.datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    traffic_level = "moderate"

    sample_data = {
        "line_id": bus_line or "L01",
        "stop_id": resolved_stop_id or "S001",
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "weather_condition": "clear",
        "traffic_level": traffic_level,
        "speed_factor": 0.85,
    }

    # If feature list exists, keep only trained columns; otherwise use all.
    model_input = (
        {feature: sample_data.get(feature) for feature in ML_FEATURES}
        if ML_FEATURES
        else sample_data
    )
    input_df = pd.DataFrame([model_input])

    prediction = ML_MODEL.predict(input_df)[0]
    eta = max(1, int(round(float(prediction))))
    explanation = "Prediction comes from the trained ML model."
    return eta, traffic_level, explanation, True


load_model()
load_bus_stops()


@app.route("/", methods=["GET", "POST"])
def home():
    eta = None
    traffic_level = None
    explanation = None
    model_used = False
    available_lines, line_stop_map = build_line_stop_options()

    bus_line = available_lines[0] if available_lines else ""
    default_stops = line_stop_map.get(bus_line, [])
    stop = default_stops[0]["stop_id"] if default_stops else ""

    if request.method == "POST":
        selected_line = request.form.get("bus_line", "").strip().upper()
        if selected_line in line_stop_map:
            bus_line = selected_line

        selected_stop = request.form.get("stop", "").strip()
        line_stops = line_stop_map.get(bus_line, [])
        stop_ids = {item["stop_id"] for item in line_stops}
        if selected_stop in stop_ids:
            stop = selected_stop
        elif line_stops:
            stop = line_stops[0]["stop_id"]

        try:
            eta, traffic_level, explanation, model_used = predict_with_model(bus_line, stop)
        except Exception:
            eta = "N/A"
            traffic_level = "moderate"
            explanation = "Prediction failed. Please retrain model and try again."
            model_used = False

    return render_template(
        "index.html",
        eta=eta,
        traffic_level=traffic_level,
        explanation=explanation,
        model_used=model_used,
        bus_line=bus_line,
        stop=stop,
        available_lines=available_lines,
        line_stop_map=line_stop_map,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
