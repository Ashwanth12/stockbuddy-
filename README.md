# StockBuddy Assistant

Interpretable and lightweight web-based decision support system for short-term
equity forecasting and recommendation, implemented from the paper:

> Sujit Shibaprasad Maity, *StockBuddy: An Interpretable and Lightweight
> Web-Based Decision Support System for Short-Term Equity Forecasting and
> Recommendation*, IJISRT, Vol. 11, Issue 3 (March 2026),
> DOI: [10.38124/ijisrt/26mar1068](https://doi.org/10.38124/ijisrt/26mar1068)

## Pipeline

1. **Data** — Alpha Vantage `TIME_SERIES_DAILY` (daily OHLCV).
2. **Forecast** — Linear Regression on lagged closes + moving averages;
   iterative forecast of the next **7 trading days** (weekends excluded).
3. **Recommendation** — transparent rule engine (BUY / HOLD / SELL) driven by
   the forecasted 7-day trend.
4. **Backtest** — rolling-window walk-forward evaluation vs a naive
   persistence baseline; MAE + directional accuracy.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

- Default API key `demo` supports the IBM symbol only.
- Get a free key at https://www.alphavantage.co/ (25 requests/day free tier).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `data.py` | Alpha Vantage data acquisition |
| `model.py` | Linear Regression forecaster (7-trading-day horizon) |
| `recommendation.py` | Rule-based recommendation engine |
| `backtest.py` | Rolling-window backtesting + naive baseline |
| `requirements.txt` | Python dependencies |

## Disclaimer

For informational/educational purposes only. Not investment advice.
Past performance does not guarantee future results.
