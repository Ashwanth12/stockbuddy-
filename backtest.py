"""Rolling-window backtesting vs a naive persistence baseline
(predict tomorrow's close = today's close), as benchmarked in the paper."""
import numpy as np
import pandas as pd

from model import StockBuddyForecaster


def backtest(close: pd.Series, train_window: int = 252, step: int = 21) -> dict:
    """Walk-forward rolling-window evaluation."""
    mae_model, mae_naive, dir_model, dir_naive = [], [], [], []

    for i in range(train_window, len(close) - 1, step):
        train = close.iloc[:i]
        actual = close.iloc[i]

        forecaster = StockBuddyForecaster().fit(train)
        pred = forecaster.predict_next_days(train, horizon=1)
        y_hat = pred["predicted_close"].iloc[0]
        y_naive = train.iloc[-1]

        mae_model.append(abs(y_hat - actual))
        mae_naive.append(abs(y_naive - actual))
        prev = train.iloc[-1]
        dir_model.append(np.sign(y_hat - prev) == np.sign(actual - prev))
        dir_naive.append(np.sign(y_naive - prev) == np.sign(actual - prev))

    return {
        "windows_evaluated": len(mae_model),
        "mae_model": float(np.mean(mae_model)),
        "mae_naive_baseline": float(np.mean(mae_naive)),
        "mae_improvement_pct": float(
            (np.mean(mae_naive) - np.mean(mae_model)) / np.mean(mae_naive) * 100),
        "directional_accuracy_model": float(np.mean(dir_model) * 100),
        "directional_accuracy_naive": float(np.mean(dir_naive) * 100),
    }