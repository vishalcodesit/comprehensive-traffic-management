"""
Cleans the UCI Metro Interstate Traffic Volume dataset, engineers time/weather
features, derives discrete congestion-level labels, builds sliding-window
sequences for multi-horizon forecasting, and caches everything to disk.

Run:  python -m src.congestion.preprocess
"""
import pickle
import sys
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def load_raw(path: str = config.METRO_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        from src.congestion.download_dataset import download
        download()
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    # The raw file has one row per (hour, weather_description) combination,
    # i.e. duplicate timestamps when multiple weather events were logged in
    # the same hour. Collapse to one row per hour.
    df = (
        df.sort_values("date_time")
        .groupby("date_time", as_index=False)
        .agg(
            {
                "holiday": lambda s: next((v for v in s if v != "None"), "None"),
                "temp": "mean",
                "rain_1h": "mean",
                "snow_1h": "mean",
                "clouds_all": "mean",
                "weather_main": "first",
                "traffic_volume": "mean",
            }
        )
    )

    # Known data-quality issue: a small number of rain_1h readings are
    # physically implausible (>1000mm in an hour). Clip to the 99.5th
    # percentile of otherwise-valid readings rather than dropping rows.
    cap = df.loc[df["rain_1h"] < 1000, "rain_1h"].quantile(0.995)
    df["rain_1h"] = df["rain_1h"].clip(upper=cap)

    # Re-index to a continuous hourly grid so the LSTM sees a uniform
    # sampling rate; interpolate short gaps, forward/back-fill the rest.
    full_idx = pd.date_range(df["date_time"].min(), df["date_time"].max(), freq="h")
    df = df.set_index("date_time").reindex(full_idx)
    df.index.name = "date_time"

    df["traffic_volume"] = df["traffic_volume"].interpolate(limit=6).bfill().ffill()
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        df[col] = df[col].interpolate(limit=6).bfill().ffill()
    df["weather_main"] = df["weather_main"].ffill().bfill()
    df["holiday"] = df["holiday"].fillna("None")

    return df.reset_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["date_time"].dt.hour
    dow = df["date_time"].dt.dayofweek
    month = df["date_time"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(float)
    df["is_holiday"] = (df["holiday"] != "None").astype(float)
    df["is_rush_hour"] = hour.isin([7, 8, 9, 16, 17, 18]).astype(float)

    # Broad weather buckets -> one-hot (rare categories collapse into "Other").
    common_weather = ["Clear", "Clouds", "Mist", "Rain", "Snow", "Drizzle", "Thunderstorm", "Fog", "Haze"]
    df["weather_bucket"] = df["weather_main"].where(df["weather_main"].isin(common_weather), "Other")
    weather_dummies = pd.get_dummies(df["weather_bucket"], prefix="wx").astype(float)

    df = pd.concat([df, weather_dummies], axis=1)
    return df, list(weather_dummies.columns)


def add_congestion_labels(df: pd.DataFrame, train_frac: float = 0.7):
    n_train = int(len(df) * train_frac)
    train_volume = df["traffic_volume"].iloc[:n_train]
    thresholds = train_volume.quantile(config.CONGESTION_QUANTILES).values

    def to_level(v):
        if v <= thresholds[0]:
            return 0  # Low
        elif v <= thresholds[1]:
            return 1  # Medium
        elif v <= thresholds[2]:
            return 2  # High
        return 3  # Severe

    df["congestion_level"] = df["traffic_volume"].apply(to_level)
    return df, thresholds


def build_windows(df: pd.DataFrame, feature_cols, lookback: int, horizons):
    """Sliding windows: X = [t-lookback .. t-1] features, y = volume at t+h-1 for each horizon h."""
    values = df[feature_cols].values.astype(np.float32)
    volume = df["traffic_volume"].values.astype(np.float32)
    level = df["congestion_level"].values.astype(np.int64)

    max_h = max(horizons)
    n = len(df)
    X, y_volume, y_level = [], [], []
    for t in range(lookback, n - max_h + 1):
        X.append(values[t - lookback:t])
        y_volume.append([volume[t + h - 1] for h in horizons])
        y_level.append([level[t + h - 1] for h in horizons])

    return (
        np.stack(X),
        np.array(y_volume, dtype=np.float32),
        np.array(y_level, dtype=np.int64),
    )


def run():
    print("Loading raw data...")
    df = load_raw()
    print(f"  {len(df)} hourly rows after cleaning/reindexing "
          f"({df['date_time'].min()} -> {df['date_time'].max()})")

    df, weather_cols = engineer_features(df)
    df, thresholds = add_congestion_labels(df)
    print(f"  Congestion thresholds (veh/hr): Low<={thresholds[0]:.0f}  "
          f"Medium<={thresholds[1]:.0f}  High<={thresholds[2]:.0f}  Severe above")

    numeric_cols = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    cyclical_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    flag_cols = ["is_weekend", "is_holiday", "is_rush_hour"]
    feature_cols = ["traffic_volume"] + numeric_cols + cyclical_cols + flag_cols + weather_cols

    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled.loc[:n_train, numeric_cols + ["traffic_volume"]] = scaler.fit_transform(
        df.loc[:n_train, numeric_cols + ["traffic_volume"]]
    )
    df_scaled.loc[n_train:, numeric_cols + ["traffic_volume"]] = scaler.transform(
        df.loc[n_train:, numeric_cols + ["traffic_volume"]]
    )
    X, _, y_level = build_windows(df_scaled, feature_cols, config.LOOKBACK_HOURS, config.HORIZONS)
    # y_volume should be the raw (unscaled) volume so RMSE/MAE are interpretable.
    volume_raw = df["traffic_volume"].values.astype(np.float32)
    max_h = max(config.HORIZONS)
    y_volume = np.array(
        [[volume_raw[t + h - 1] for h in config.HORIZONS]
         for t in range(config.LOOKBACK_HOURS, n - max_h + 1)],
        dtype=np.float32,
    )

    split_a = n_train - config.LOOKBACK_HOURS
    split_b = n_val - config.LOOKBACK_HOURS

    np.savez_compressed(
        config.PROCESSED_NPZ,
        X_train=X[:split_a], y_train=y_volume[:split_a], lvl_train=y_level[:split_a],
        X_val=X[split_a:split_b], y_val=y_volume[split_a:split_b], lvl_val=y_level[split_a:split_b],
        X_test=X[split_b:], y_test=y_volume[split_b:], lvl_test=y_level[split_b:],
        feature_cols=np.array(feature_cols),
        horizons=np.array(config.HORIZONS),
        congestion_thresholds=thresholds,
    )
    with open(config.SCALERS_PKL, "wb") as f:
        pickle.dump({"scaler": scaler, "numeric_cols": numeric_cols,
                     "feature_cols": feature_cols, "thresholds": thresholds}, f)

    print(f"Saved {X.shape[0]} windows -> {config.PROCESSED_NPZ}")
    print(f"  train={split_a}  val={split_b - split_a}  test={X.shape[0] - split_b}")
    print(f"  feature dim={X.shape[2]}  lookback={config.LOOKBACK_HOURS}  horizons={config.HORIZONS}")


if __name__ == "__main__":
    run()
