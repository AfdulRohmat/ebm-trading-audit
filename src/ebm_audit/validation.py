import numpy as np
import pandas as pd

def audit_trade(trade, part):
    """Independent vectorized stop/market-fill check, including trail history."""
    side, entry, slip, fee = [trade[k] for k in ["side", "entry_price", "exit_slippage", "commission_points"]]
    stress, risk = trade["stress"], trade["risk_points"]
    quotes = {}
    for field in ["open", "high", "low", "close"]:
        b, a = part[f"bid_{field}"].to_numpy(), part[f"ask_{field}"].to_numpy()
        quotes[field] = (a + b) / 2 - side * stress * (a - b) / 2
    times = pd.DatetimeIndex(part.timestamp)
    end = times[-1] + pd.Timedelta(minutes=1)
    # Independently derive every eligible completed-close stop change, without
    # calling the state-machine implementation or trusting its reported stops.
    expected, best_stop = [], trade["initial_stop"]
    if trade["variant"] == "runner_stop1":
        closes = times + pd.Timedelta(minutes=1)
        eligible = (closes.minute % 30 == 0) & (closes < end) & (closes <= trade["exit_time"])
        close_r = (side * (quotes["close"][eligible] - entry) - slip - fee) / risk
        best = np.maximum.accumulate(close_r)
        for timestamp, best_r in zip(closes[eligible], best):
            if best_r >= 1:
                candidate = entry + side * ((best_r - 1) * risk + slip + fee)
                if side * (candidate - best_stop) > 0:
                    expected.append((timestamp, candidate))
                    best_stop = candidate
    actual = [(r["effective_open_time"], r["new_stop"]) for r in trade["_trace"]]
    assert len(expected) == len(actual)
    for (at, ap), (bt, bp) in zip(expected, actual):
        assert at == bt
        assert abs(ap - bp) < 1e-10
    active = np.full(len(times), trade["initial_stop"])
    for timestamp, stop in expected:
        active[times >= timestamp] = stop
    adverse = quotes["low"] if side == 1 else quotes["high"]
    stop_hits = np.flatnonzero(side * (adverse - active) <= 0)
    stop_index = int(stop_hits[0]) if len(stop_hits) else len(times)
    if trade["variant"].startswith("fixed60"):
        market_time, market_reason = trade["entry_time"] + pd.Timedelta(minutes=60), "horizon"
    elif trade["opposite_time"] is not None:
        market_time, market_reason = trade["opposite_time"], "opposite"
    else:
        market_time, market_reason = end, "session_close"
    market_index = int(times.searchsorted(market_time))
    gap_tie = (
        stop_index == market_index
        and stop_index < len(times)
        and (side * (quotes["open"][stop_index] - active[stop_index]) <= 0)
    )
    if stop_index < market_index or gap_tie:
        k = stop_index
        q = min(active[k], quotes["open"][k]) if side == 1 else max(active[k], quotes["open"][k])
        exit_time = times[k]
        reason = "trailing_stop" if side * (active[k] - trade["initial_stop"]) > 0 else "initial_stop"
    elif market_index < len(times):
        q, exit_time, reason = quotes["open"][market_index], times[market_index], market_reason
    else:
        q, exit_time, reason = quotes["close"][-1], end, "session_close"
    assert trade["exit_time"] == exit_time
    assert trade["exit_reason"] == reason
    error = abs(trade["exit_price"] - (q - side * slip))
    assert error < 1e-10
    assert abs(trade["net_points"] - (side * (trade["exit_price"] - entry) - fee)) < 1e-10
    assert abs(trade["net_r"] * risk - trade["net_points"]) < 1e-10
    return error
