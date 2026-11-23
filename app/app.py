import os, sys
import joblib, pandas as pd
from flask import Flask, request, jsonify, render_template

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, ".."))
from src.predict import predict_next_day, bucket

bundle = joblib.load(os.path.join(BASE, "..", "models", "aqi_bundle.joblib"))
history = pd.read_csv(os.path.join(BASE, "..", "data", "processed", "history.csv"),
                      parse_dates=["Date"])
app = Flask(__name__)

DATA_END = history["Date"].max()
CITY_RANGE = history.groupby("City")["Date"].agg(["min", "max"])
NOTE = "Forecasts are based on historical data, not live AQI."

def freshness(age):
    return "Fresh" if age <= 1 else "Stale" if age <= 7 else "Very stale"

def make_forecast(city, as_of):
    pred, last_aqi = predict_next_day(history, bundle, city, as_of)
    age = (DATA_END - as_of).days
    out = {"city": city, "as_of": str(as_of.date()),
           "forecast_for": str((as_of + pd.Timedelta(days=1)).date()),
           "predicted_aqi": round(pred), "category": bucket(pred),
           "data_age_days": age, "freshness": freshness(age),
           "last_observed_aqi": None if pd.isna(last_aqi) else round(float(last_aqi)),}
    if pd.isna(last_aqi):
        out["warning"] = "AQI for the as-of date is missing, so this forecast is less reliable."
    return out

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"message": "AirSight India API is running!"})

@app.route("/cities")
def cities():
    return jsonify({c: {"from": str(r["min"].date()), "to": str(r["max"].date())}
                    for c, r in CITY_RANGE.iterrows()})

@app.route("/forecasts")
def forecasts():
    rows = [make_forecast(c, r["max"]) for c, r in CITY_RANGE.iterrows()]
    return jsonify({"data_through": str(DATA_END.date()), "note": NOTE, "predictions": rows})

@app.route("/predict")
def predict():
    city = request.args.get("city")
    if city not in CITY_RANGE.index:
        return jsonify({"error": "Unknown city"}), 400
    try:
        as_of = pd.Timestamp(request.args.get("date", CITY_RANGE.loc[city, "max"]))
    except ValueError:
        return jsonify({"error": "Bad date, use YYYY-MM-DD"}), 400
    if as_of > CITY_RANGE.loc[city, "max"] or as_of < CITY_RANGE.loc[city, "min"]:
        return jsonify({"error": "Date outside this city's data range"}), 400
    return jsonify(make_forecast(city, as_of))

@app.route("/trend")
def trend():
    city = request.args.get("city")
    if city not in CITY_RANGE.index:
        return jsonify({"error": "Unknown city"}), 400
    end = pd.Timestamp(request.args.get("end", CITY_RANGE.loc[city, "max"]))
    days = int(request.args.get("days", 90))
    h = history[(history["City"] == city) & (history["Date"] <= end)].sort_values("Date").tail(days)
    return jsonify({"dates": [str(d.date()) for d in h["Date"]],
                    "aqi": [None if pd.isna(v) else round(float(v), 1) for v in h["AQI"]]})

if __name__ == "__main__":
    app.run(debug=True)