"""
Uber Trip Analytics & Demand Insights
--------------------------------------
Exploratory Data Analysis (EDA) of NYC Uber pickup data (Apr-Sep 2014)
using pandas, NumPy, Matplotlib, and Seaborn.

Run with:  python uber_analysis.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = ["#CC1011", "#665555", "#05a399", "#cfcaca",
           "#f5e840", "#0683c9", "#e075b0"]

# NYC bounding box (used to crop the geospatial scatter plot)
MIN_LAT, MAX_LAT = 40.5774, 40.9176
MIN_LON, MAX_LON = -74.15, -73.7004


def save(fig, name):
    """Save a matplotlib figure to outputs/plots/ and close it."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------
# 1. Load and combine the six monthly CSV files
# ---------------------------------------------------------------
def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "uber-raw-data-*14.csv")))
    if not files:
        raise FileNotFoundError(
            "No Uber CSV files found in data/. Place the 6 CSVs there first."
        )
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} files, {len(df):,} total rows.")
    return df


# ---------------------------------------------------------------
# 2. Feature engineering: parse Date/Time, extract time components
# ---------------------------------------------------------------
def add_time_features(df):
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%m/%d/%Y %H:%M:%S")

    df["hour"] = df["Date/Time"].dt.hour
    df["minute"] = df["Date/Time"].dt.minute
    df["second"] = df["Date/Time"].dt.second
    df["day"] = df["Date/Time"].dt.day
    df["month"] = df["Date/Time"].dt.month_name().str[:3]     # Apr, May, ...
    df["year"] = df["Date/Time"].dt.year
    df["dayofweek"] = df["Date/Time"].dt.day_name().str[:3]   # Mon, Tue, ...

    # Keep month/weekday in calendar order (not alphabetical) for plotting
    month_order = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
    df["dayofweek"] = pd.Categorical(df["dayofweek"], categories=weekday_order, ordered=True)

    return df


# ---------------------------------------------------------------
# 3. Trips by hour of day
# ---------------------------------------------------------------
def plot_trips_by_hour(df):
    hour_data = df.groupby("hour").size().reset_index(name="Total")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hour_data["hour"], hour_data["Total"], color="steelblue", edgecolor="red")
    ax.set_title("Trips Every Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total Trips")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x):,}")
    save(fig, "01_trips_by_hour")

    month_hour = df.groupby(["month", "hour"], observed=True).size().unstack(0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    month_hour.plot(kind="bar", stacked=True, ax=ax, color=PALETTE)
    ax.set_title("Trips by Hour and Month")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total Trips")
    ax.legend(title="Month")
    save(fig, "02_trips_by_hour_and_month")


# ---------------------------------------------------------------
# 4. Trips by day of the month
# ---------------------------------------------------------------
def plot_trips_by_day(df):
    day_group = df.groupby("day").size().reset_index(name="Total")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(day_group["day"], day_group["Total"], color="steelblue")
    ax.set_title("Trips Every Day")
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Total Trips")
    save(fig, "03_trips_by_day")

    day_month = df.groupby(["month", "day"], observed=True).size().unstack(0).fillna(0)
    fig, ax = plt.subplots(figsize=(14, 6))
    day_month.plot(kind="bar", stacked=True, ax=ax, color=PALETTE)
    ax.set_title("Trips by Day and Month")
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Total Trips")
    ax.legend(title="Month")
    save(fig, "04_trips_by_day_and_month")


# ---------------------------------------------------------------
# 5. Trips by month, and by month + weekday
# ---------------------------------------------------------------
def plot_trips_by_month(df):
    month_group = df.groupby("month", observed=True).size().reset_index(name="Total")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(month_group["month"].astype(str), month_group["Total"], color=PALETTE[:len(month_group)])
    ax.set_title("Trips by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Trips")
    save(fig, "05_trips_by_month")

    month_weekday = (
        df.groupby(["month", "dayofweek"], observed=True)
        .size()
        .reset_index(name="Total")
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=month_weekday, x="month", y="Total", hue="dayofweek",
                palette=PALETTE, ax=ax)
    ax.set_title("Trips by Day of Week and Month")
    save(fig, "06_trips_by_month_and_weekday")


# ---------------------------------------------------------------
# 6. Trips by dispatch Base
# ---------------------------------------------------------------
def plot_trips_by_base(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Base"].value_counts().plot(kind="bar", color="darkred", ax=ax)
    ax.set_title("Trips by Base")
    ax.set_xlabel("Base")
    ax.set_ylabel("Total Trips")
    save(fig, "07_trips_by_base")

    base_month = df.groupby(["Base", "month"], observed=True).size().reset_index(name="Total")
    n_months = base_month["month"].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=base_month, x="Base", y="Total", hue="month", palette=PALETTE[:n_months], ax=ax)
    ax.set_title("Trips by Base and Month")
    save(fig, "08_trips_by_base_and_month")

    base_weekday = df.groupby(["Base", "dayofweek"], observed=True).size().reset_index(name="Total")
    n_days = base_weekday["dayofweek"].nunique()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=base_weekday, x="Base", y="Total", hue="dayofweek", palette=PALETTE[:n_days], ax=ax)
    ax.set_title("Trips by Base and Day of Week")
    save(fig, "09_trips_by_base_and_weekday")


# ---------------------------------------------------------------
# 7. Heatmaps: cross-tabulations of two categorical dimensions
# ---------------------------------------------------------------
def plot_heatmaps(df):
    pivot = df.groupby(["day", "hour"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    ax.set_title("Heat Map by Hour and Day")
    save(fig, "10_heatmap_hour_day")

    pivot = df.groupby(["month", "day"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    ax.set_title("Heat Map by Month and Day")
    save(fig, "11_heatmap_month_day")

    pivot = df.groupby(["month", "dayofweek"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Heat Map by Month and Day of Week")
    save(fig, "12_heatmap_month_weekday")

    pivot = df.groupby(["Base", "month"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Heat Map by Base and Month")
    save(fig, "13_heatmap_base_month")

    pivot = df.groupby(["Base", "dayofweek"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Heat Map by Base and Day of Week")
    save(fig, "14_heatmap_base_weekday")


# ---------------------------------------------------------------
# 8. Geospatial scatter plot of pickups across NYC
# ---------------------------------------------------------------
def plot_geo(df):
    mask = (
        df["Lat"].between(MIN_LAT, MAX_LAT)
        & df["Lon"].between(MIN_LON, MAX_LON)
    )
    geo_df = df.loc[mask]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(geo_df["Lon"], geo_df["Lat"], s=1, c="blue", alpha=0.3)
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_title("NYC Map Based on Uber Rides During 2014 (Apr-Sep)")
    ax.axis("off")
    save(fig, "15_nyc_map_all_rides")

    fig, ax = plt.subplots(figsize=(10, 10))
    for base, color in zip(sorted(geo_df["Base"].unique()), PALETTE):
        subset = geo_df[geo_df["Base"] == base]
        ax.scatter(subset["Lon"], subset["Lat"], s=1, c=color, alpha=0.4, label=base)
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_title("NYC Map Based on Uber Rides During 2014 (Apr-Sep) by Base")
    ax.legend(markerscale=10, loc="upper right")
    ax.axis("off")
    save(fig, "16_nyc_map_by_base")


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def main():
    df = load_data()
    df = add_time_features(df)

    plot_trips_by_hour(df)
    plot_trips_by_day(df)
    plot_trips_by_month(df)
    plot_trips_by_base(df)
    plot_heatmaps(df)
    plot_geo(df)

    print("\nAll plots generated successfully in outputs/plots/")


if __name__ == "__main__":
    main()