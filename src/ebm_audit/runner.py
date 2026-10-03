"""Causal, completed-M30-close ATR runner; no changes to frozen v1/v2 code."""

import numpy as np
import pandas as pd

from .trading import commission_unit, slippage


def complete_session_path(event, m1, session_close):
    """Explicit retrospective data-availability check, not a trading filter."""
    start = int(event.start_idx)
    expected = pd.date_range(event.entry_time, session_close, freq="min", inclusive="left")
    part = m1.iloc[start : start + len(expected)]
    if len(expected) == 0 or not pd.DatetimeIndex(part.timestamp).equals(expected):
        return None
    return part


def simulate(event, part, side, spec, variant, config, stress=1, opposite_time=None):
    if variant not in config["variants"] or side not in (-1, 1):
        raise ValueError("Unknown variant/direction")
    if part.empty or part.timestamp.iloc[0] != event.entry_time:
        raise ValueError("Execution path must start at entry")
    if len(part) > 1 and not part.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Execution path must be complete M1")
    quotes = {}
    for field in ["open", "high", "low", "close"]:
        bid, ask = part[f"bid_{field}"].to_numpy(), part[f"ask_{field}"].to_numpy()
        mid, half = (bid + ask) / 2, stress * (ask - bid) / 2
        quotes[field] = mid - half if side == 1 else mid + half
    entry_mid = float(event.entry_mid)
    slip, fee = float(slippage(entry_mid, spec, stress)), commission_unit(spec)
    entry = entry_mid + side * (float(event.entry_spread) * stress / 2 + slip)
    atr = float(event.atr)
    distance = (2 if variant == "fixed60_stop2" else 1) * atr
    risk_points = distance + slip + fee
    if risk_points <= 0 or not np.isfinite(risk_points):
        raise ValueError("Invalid risk distance")
    stop = entry - side * distance
    horizon = event.entry_time + pd.Timedelta(minutes=60)
    runner = variant == "runner_stop1"
    best_r = -np.inf
    path, trace = [], []
    trail_active = False
    exit_quote = exit_time = reason = None
    initial_stop = stop
    for k, timestamp in enumerate(part.timestamp):
        op, hi, lo, cl = [quotes[field][k] for field in ["open", "high", "low", "close"]]
        adverse, favorable = (lo, hi) if side == 1 else (hi, lo)
        # At an open, a pre-existing stop gap executes before a market exit.
        if side * (op - stop) <= 0:
            exit_quote, exit_time = op, timestamp
            reason = "trailing_stop" if trail_active else "initial_stop"
            break
        if not runner and timestamp >= horizon:
            exit_quote, exit_time, reason = op, timestamp, "horizon"
            break
        if runner and opposite_time is not None and timestamp >= opposite_time:
            exit_quote, exit_time, reason = op, timestamp, "opposite"
            break
        # Include the executable open. Do not count favorable extremes on the
        # stop minute: OHLC does not tell whether they preceded the stop.
        path.append(float(side * (op - entry) - slip - fee))
        if side * (adverse - stop) <= 0:
            exit_quote, exit_time = stop, timestamp
            reason = "trailing_stop" if trail_active else "initial_stop"
            break
        path.extend(float(side * (q - entry) - slip - fee) for q in [favorable, adverse, cl])
        close_time = timestamp + pd.Timedelta(minutes=1)
        if k == len(part) - 1:
            exit_quote, exit_time, reason = cl, close_time, "session_close"
            break
        if runner and close_time.minute % config["trail_clock_minutes"] == 0:
            net_r = (side * (cl - entry) - slip - fee) / risk_points
            best_r = max(best_r, net_r)
            if best_r >= config["trail_activation_r"]:
                locked_r = best_r - config["trail_distance_r"]
                candidate = entry + side * (locked_r * risk_points + slip + fee)
                if side * (candidate - stop) > 0:
                    trace.append(
                        {
                            "observed_close_time": close_time,
                            "effective_open_time": part.timestamp.iloc[k + 1],
                            "old_stop": float(stop),
                            "new_stop": float(candidate),
                            "best_close_r": float(best_r),
                            "locked_r": float(locked_r),
                        }
                    )
                    stop, trail_active = candidate, True
    if exit_quote is None:
        raise AssertionError("Unclosed position")
    exit_price = exit_quote - side * slip
    net = side * (exit_price - entry) - fee
    path.append(float(net))
    return {
        "entry_time": event.entry_time,
        "exit_time": exit_time,
        "day": event.day,
        "side": side,
        "entry_price": entry,
        "exit_price": float(exit_price),
        "atr": atr,
        "initial_stop": float(initial_stop),
        "final_stop": float(stop),
        "stop_distance": distance,
        "risk_points": risk_points,
        "exit_slippage": slip,
        "commission_points": fee,
        "net_points": float(net),
        "net_r": float(net / risk_points),
        "nominal_stop_r": float(net / distance),
        "net_atr": float(net / atr),
        "mfe_r": float(max(0, max(path)) / risk_points),
        "mae_r": float(max(0, -min(path)) / risk_points),
        "exit_reason": reason,
        "stress": stress,
        "variant": variant,
        "trail_activated": trail_active,
        "trail_updates": len(trace),
        "holding_minutes": (exit_time - event.entry_time).total_seconds() / 60,
        "opposite_time": opposite_time,
        "_path": path,
        "_trace": trace,
    }


def account_scenario(trades, capital, risk, all_months):
    """Ideal fractional units, risk is exact planned stop loss, not notional."""
    if capital <= 0 or not 0 < risk < 1:
        raise ValueError("Invalid capital/risk")
    balance = peak_balance = peak_marked = float(capital)
    balance_dd = marked_dd = 0.0
    min_equity = float(capital)
    max_loss_budget = 0.0
    insolvent = False
    rows = []
    for trade in trades.to_dict("records"):
        if balance <= 0:
            insolvent = True
            break
        before, budget = balance, balance * risk
        units = budget / trade["risk_points"]
        equity = before + units * np.asarray(trade["_path"])
        nonpositive = np.flatnonzero(equity <= 0)
        if len(nonpositive):
            equity = equity[: nonpositive[0] + 1]
            insolvent = True
        peaks = np.maximum.accumulate(np.r_[peak_marked, equity])[1:]
        marked_dd = max(marked_dd, float(((peaks - equity) / peaks).max()))
        peak_marked = max(peak_marked, float(equity.max()))
        min_equity = min(min_equity, float(equity.min()))
        # If insolvent, truncate at first modeled nonpositive mark. This is NOT
        # an actual broker liquidation/stop-out price or negative-balance rule.
        pnl = float(equity[-1] - before) if insolvent else units * trade["net_points"]
        balance = before + pnl
        max_loss_budget = max(max_loss_budget, -pnl / budget)
        peak_balance = max(peak_balance, balance)
        balance_dd = max(balance_dd, (peak_balance - balance) / peak_balance)
        rows.append(
            {
                "day": trade["day"],
                "entry_time": trade["entry_time"],
                "exit_time": trade["exit_time"],
                "balance_before": before,
                "risk_budget": budget,
                "units": units,
                "notional_leverage": units * trade["entry_price"] / before,
                "pnl": pnl,
                "balance": balance,
            }
        )
        if insolvent:
            break
    history = pd.DataFrame(rows)
    if len(history):
        month_end = history.groupby(history.day.str[:7]).balance.last().reindex(all_months).ffill()
        month_end = month_end.fillna(capital)
        monthly = month_end / month_end.shift(fill_value=capital) - 1
        max_leverage = float(history.notional_leverage.max())
        worst_month = float(monthly.min() * 100)
    else:
        worst_month, max_leverage = 0.0, 0.0
    years = len(all_months) / 12
    return {
        "capital": capital,
        "risk_pct": risk * 100,
        "executed": len(rows),
        "final_balance": balance,
        "return_pct": (balance / capital - 1) * 100,
        "annualized_return_pct": ((balance / capital) ** (1 / years) - 1) * 100
        if balance > 0 and years > 0
        else None,
        "balance_dd_pct": balance_dd * 100,
        "modeled_marked_dd_pct": marked_dd * 100,
        "min_equity": min_equity,
        "max_loss_to_budget": max_loss_budget,
        "worst_month_pct": worst_month,
        "max_notional_leverage": max_leverage,
        "insolvent": insolvent,
        "sizing_type": "ideal_fractional_SPY_not_Exness_no_margin_check",
    }, history


def paired_delta_bootstrap(base, alternative, days, seed, draws=2000):
    """Use ATR, not each variant's different R denominator, for paired change."""
    if base.entry_time.tolist() != alternative.entry_time.tolist():
        raise AssertionError("Unpaired entries")
    delta = alternative.net_atr.to_numpy() - base.net_atr.to_numpy()
    frame = pd.DataFrame({"day": base.day, "delta": delta})
    daily = frame.groupby("day").delta.agg(["sum", "count"]).reindex(days, fill_value=0)
    sums, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    rng, samples = np.random.default_rng(seed), []
    for _ in range(draws):
        starts = rng.integers(0, len(days), size=(len(days) + 4) // 5)
        idx = ((starts[:, None] + np.arange(5)) % len(days)).ravel()[: len(days)]
        if counts[idx].sum():
            samples.append(sums[idx].sum() / counts[idx].sum())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_delta_atr": float(delta.mean()),
        "total_delta_atr": float(delta.sum()),
        "ci_low_delta_atr": float(low),
        "ci_high_delta_atr": float(high),
    }
