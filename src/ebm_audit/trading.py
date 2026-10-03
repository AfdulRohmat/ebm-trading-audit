import numpy as np

def slippage(price, spec, stress=1):
    return stress * (spec.get("slippage_point_side", 0) + price * spec.get("slippage_bps_side", 0) / 10000)


def commission_unit(spec):
    return 2 * spec["commission_side_lot"] / spec["contract"]


def directions(events, predictions, spec, threshold=0.05, stress=1):
    cost = (
        events.entry_spread.to_numpy() * stress
        + 2 * slippage(events.entry_mid.to_numpy(), spec, stress)
        + commission_unit(spec)
    )
    hurdle = cost / events.atr.to_numpy() + threshold
    p = np.asarray(predictions)
    return np.where(p > hurdle, 1, np.where(p < -hurdle, -1, 0))


def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "net_r": 0.0,
            "expectancy_r": None,
            "pf": None,
            "win_rate": None,
            "max_loss_streak": 0,
            "remove_top2_net_r": 0.0,
            "trade_equity_dd_r": 0.0,
            "stop_rate": None,
        }
    r = trades.net_r.to_numpy()
    losses, wins = -r[r < 0].sum(), r[r > 0].sum()
    streak = maximum = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        maximum = max(maximum, streak)
    eq = np.r_[0, r.cumsum()]
    return {
        "trades": len(r),
        "net_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "pf": float(wins / losses) if losses else None,
        "win_rate": float((r > 0).mean()),
        "max_loss_streak": maximum,
        "remove_top2_net_r": float(np.sort(r)[:-2].sum()) if len(r) >= 2 else 0.0,
        "trade_equity_dd_r": float((np.maximum.accumulate(eq) - eq).max()),
        "stop_rate": float(trades.exit_reason.eq("stop").mean()),
    }
