"""Rule-based recommendation engine - transparent guidance from the
forecasted trend (no black box), per the paper's philosophy."""


def recommend(last_close: float, forecast_df) -> dict:
    """Map forecasted trend to a recommendation using transparent thresholds."""
    fc = forecast_df["predicted_close"]
    expected = float(fc.iloc[-1])
    change_pct = (expected - last_close) / last_close * 100
    swing_pct = (float(fc.max()) - float(fc.min())) / last_close * 100

    if change_pct >= 2.0:
        action, reason = "BUY / ACCUMULATE", (
            f"Forecast projects +{change_pct:.2f}% over the 7-trading-day horizon "
            f"(last close {last_close:.2f} -> projected {expected:.2f}).")
    elif change_pct <= -2.0:
        action, reason = "SELL / AVOID ENTRY", (
            f"Forecast projects {change_pct:.2f}% decline "
            f"(last close {last_close:.2f} -> projected {expected:.2f}).")
    else:
        action, reason = "HOLD / WATCH", (
            f"Forecast is roughly flat ({change_pct:+.2f}%); no clear edge within "
            "the horizon. Re-evaluate on the next data update.")

    volatility_note = (
        f"Projected path is volatile (intra-horizon swing {swing_pct:.2f}%); "
        "linear models weaken in highly volatile conditions."
        if swing_pct > 5.0 else
        f"Projected path is relatively stable (swing {swing_pct:.2f}%).")

    return {
        "action": action,
        "expected_price": round(expected, 2),
        "expected_change_pct": round(change_pct, 2),
        "reason": reason,
        "volatility_note": volatility_note,
    }