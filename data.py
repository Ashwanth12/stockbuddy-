"""Data acquisition module - Alpha Vantage daily time series."""
import requests
import pandas as pd

BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily_history(symbol: str, api_key: str = "demo") -> pd.DataFrame:
    """Retrieve recent daily OHLCV history (last ~100 trading days).

    Note: outputsize="full" (20+ years) is a premium Alpha Vantage feature,
    so the free tier uses "compact". ~100 rows covers the lag/MA features
    and a short rolling-window backtest. Raises ValueError with Alpha
    Vantage's own message on rate limits, invalid keys, or bad symbols.
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol.upper(),
        "outputsize": "compact",
        "datatype": "json",
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    key = "Time Series (Daily)"
    if key not in payload:
        raise ValueError(payload.get("Note") or payload.get("Information")
                         or payload.get("Error Message", "Unknown API error"))

    df = pd.DataFrame(payload[key]).T.astype(float)
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. volume": "volume",
    })
    df.index = pd.to_datetime(df.index)
    return df.sort_index()