
import pandas as pd


def build_features(df, keep_cities, lags=(1, 2, 3)):
    d = df[df["City"].isin(keep_cities)].sort_values(
        ["City", "Date"]
    ).copy()

    g = d.groupby("City")

    pollutants = [
        c for c in d.columns
        if c not in ["City", "Date", "AQI"]
    ]

    new = {}

    for l in lags:
        new[f"AQI_lag{l}"] = g["AQI"].shift(l)

    for p in pollutants:
        new[f"{p}_lag1"] = g[p].shift(1)

    new["AQI_roll7"] = g["AQI"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=4).mean()
    )

    new["PM2.5_roll7"] = g["PM2.5"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=4).mean()
    )

    d = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)

    d["month"] = d["Date"].dt.month
    d["dayofweek"] = d["Date"].dt.day_of_week

    d = d.drop(columns=pollutants)
    d = d.dropna(subset=["AQI", "AQI_lag1"])

    d["City"] = d["City"].astype("category")

    return d.reset_index(drop=True)
