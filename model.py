"""Forecasting module - interpretable Linear Regression, 7-trading-day horizon.

Pipeline (per the paper):
  1. Trading-day index with weekends excluded.
  2. Supervised features: lagged closes + moving averages.
  3. Fit OLS Linear Regression, iteratively forecast next 7 trading days.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FORECAST_HORIZON = 7
LAGS = [1, 2, 3, 5, 7, 14]
WINDOWS = [3, 7, 14]
FEATURE_NAMES = [f"lag_{l}" for l in LAGS] + [f"ma_{w}" for w in WINDOWS]


def build_features(close: pd.Series):
    """Feature matrix X (lagged closes + MAs) and target y (next-day close)."""
    df = close.to_frame("close")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    for w in WINDOWS:
        df[f"ma_{w}"] = df["close"].shift(1).rolling(w).mean()
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    return df[FEATURE_NAMES], df["target"]


def next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    """Advance one day, skipping weekends (trading-day constraint)."""
    d = d + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


class StockBuddyForecaster:
    """Lightweight, interpretable forecaster with sub-millisecond inference."""

    def __init__(self):
        self.model = LinearRegression()
        self.last_inference_ms = None

    def fit(self, close: pd.Series):
        X, y = build_features(close)
        self.model.fit(X.values, y.values)
        return self

    def predict_next_days(self, close: pd.Series,
                          horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
        """Iteratively forecast the next `horizon` trading-day closes."""
        preds, dates = [], []
        series = close.copy()
        last_date = series.index[-1]

        t0 = time.perf_counter()
        for _ in range(horizon):
            row = {}
            for lag in LAGS:
                row[f"lag_{lag}"] = series.iloc[-lag]
            for w in WINDOWS:
                row[f"ma_{w}"] = series.iloc[-w:].mean()
            X = np.array([[row[c] for c in FEATURE_NAMES]])
            y_hat = float(self.model.predict(X)[0])
            last_date = next_trading_day(last_date)
            preds.append(y_hat)
            dates.append(last_date)
            series = pd.concat([series, pd.Series({last_date: y_hat})])
        self.last_inference_ms = (time.perf_counter() - t0) * 1000

        return pd.DataFrame({"predicted_close": preds},
                            index=pd.DatetimeIndex(dates))

    @property
    def coefficients(self) -> pd.Series:
        """Interpretability: expose learned feature weights."""
        return pd.Series(self.model.coef_, index=FEATURE_NAMES)