import numpy as np
import pandas as pd

BINS = [0, 50, 100, 200, 300, 400, 500]
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

def bucket(aqi):
    return str(pd.cut([aqi], bins=BINS, labels=LABELS, include_lowest=True)[0])

def predict_next_day(history, bundle, city, as_of):
    as_of = pd.Timestamp(as_of)
    h = history[(history["City"] == city) & (history["Date"] <= as_of)]
    h = h.sort_values("Date").set_index("Date")
    day = pd.Timedelta(days=1)

    def val(col, d):
        return h[col].get(d, np.nan)

    def roll7(col):
        s = h[col].loc[as_of - 6 * day: as_of]
        return s.mean() if s.notna().sum() >= 4 else np.nan

    row = {"City": city,
           "AQI_lag1": val("AQI", as_of),
           "AQI_lag2": val("AQI", as_of - day),
           "AQI_lag3": val("AQI", as_of - 2 * day),
           "AQI_roll7": roll7("AQI"),
           "PM2.5_roll7": roll7("PM2.5"),
           "month": (as_of + day).month,
           "dayofweek": (as_of + day).dayofweek}
    for f in bundle["features"]:
        if f.endswith("_lag1") and f != "AQI_lag1":
            row[f] = val(f[:-5], as_of)

    X = pd.DataFrame([row])[bundle["features"]]
    X["City"] = pd.Categorical(X["City"], categories=bundle["cities"])
    pred = float(bundle["model"].predict(X)[0])
    return min(max(pred, 0), 500), row["AQI_lag1"]