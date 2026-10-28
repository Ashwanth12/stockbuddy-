"""StockBuddy Assistant - Streamlit UI (data -> visualize -> forecast -> recommend)."""
import streamlit as st
import pandas as pd

from data import fetch_daily_history
from model import StockBuddyForecaster
from recommendation import recommend
from backtest import backtest

st.set_page_config(page_title="StockBuddy Assistant", layout="wide")
st.title("📈 StockBuddy Assistant")
st.caption(
    "Interpretable & lightweight decision support for short-term equity "
    "forecasting - Linear Regression, 7-trading-day horizon, rule-based advice. "
    "Educational tool; **not investment advice**.")

with st.sidebar:
    symbol = st.text_input("Stock symbol (e.g., IBM, AAPL)", "IBM")
    api_key = st.text_input("Alpha Vantage API key", "demo",
                            help="Free keys at alphavantage.co (demo key only supports IBM).")
    run = st.button("Analyze")

if run:
    try:
        with st.spinner("Fetching daily history..."):
            df = fetch_daily_history(symbol, api_key)
    except ValueError as e:
        st.error(f"Data error: {e}")
        st.stop()

    close = df["close"]
    st.subheader(f"{symbol.upper()} - last close: {close.iloc[-1]:.2f} "
                 f"({df.index[-1].date()})")

    tab1, tab2, tab3 = st.tabs(["Price History", "7-Day Forecast",
                                "Backtest & Recommendation"])

    with tab1:
        st.line_chart(close.tail(500))
        st.dataframe(df.tail(10))

    with tab2:
        forecaster = StockBuddyForecaster().fit(close)
        fc = forecaster.predict_next_days(close)
        st.line_chart(pd.concat([close.tail(120), fc["predicted_close"]]))
        st.write(f"Inference latency: **{forecaster.last_inference_ms:.4f} ms**")
        st.markdown("**Learned model coefficients (interpretability)**")
        st.bar_chart(forecaster.coefficients)

    with tab3:
        rec = recommend(float(close.iloc[-1]), fc)
        m = backtest(close)
        col1, col2, col3 = st.columns(3)
        col1.metric("Recommendation", rec["action"])
        col2.metric("Expected price (7d)", rec["expected_price"],
                    f"{rec['expected_change_pct']:+.2f}%")
        col3.metric("Model MAE vs Naive", f"{m['mae_model']:.2f}",
                    f"{m['mae_improvement_pct']:+.1f}% better")
        st.success(rec["reason"])
        st.info(rec["volatility_note"])
        st.markdown(
            f"""**Rolling-window backtest** ({m['windows_evaluated']} windows)

| Metric | StockBuddy (LR) | Naive baseline |
|---|---|---|
| MAE | {m['mae_model']:.3f} | {m['mae_naive_baseline']:.3f} |
| Directional accuracy | {m['directional_accuracy_model']:.1f}% | {m['directional_accuracy_naive']:.1f}% |""")
        st.warning("Linear models capture short-term direction under stable "
                   "conditions but remain limited in highly volatile markets.")