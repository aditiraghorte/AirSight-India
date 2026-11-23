import pandas as pd

def clean_data(df, drop_cols=("Xylene",)):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.drop(columns=list(drop_cols), errors="ignore")

    df = df.sort_values(["City", "Date"])
    df = df.drop_duplicates(["City", "Date"])

    # Make missing calendar dates explicit for every city
    full = []

    for city, group in df.groupby("City"):
        dates = pd.date_range(
            group["Date"].min(),
            group["Date"].max(),
            freq="D"
        )

        group = group.set_index("Date").reindex(dates)
        group["City"] = city

        full.append(
            group.rename_axis("Date").reset_index()
        )

    df = pd.concat(full, ignore_index=True)

    # Remove impossible AQI target values
    df.loc[df["AQI"] > 500, "AQI"] = float("nan")

    # Fill pollutant gaps using past observations only
    pollutants = [
        col for col in df.columns
        if col not in ["City", "Date", "AQI", "AQI_Bucket"]
    ]

    df[pollutants] = (
        df.groupby("City")[pollutants]
        .transform(lambda x: x.ffill(limit=3))
    )

    # AQI_Bucket will be recreated later for classification
    df = df.drop(columns=["AQI_Bucket"], errors="ignore")

    return df.reset_index(drop=True)
