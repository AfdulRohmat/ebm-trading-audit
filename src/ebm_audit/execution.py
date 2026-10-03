"""Validated execution acceleration and exact compressed marked-DD summaries."""

import ctypes

import numpy as np
import pandas as pd

from .trading import commission_unit, slippage

QUOTE_COLUMNS = [f"{side}_{field}" for side in ["bid", "ask"] for field in ["open", "high", "low", "close"]]
REASONS = ["initial_stop", "trailing_stop", "horizon", "opposite", "session_close"]


class Executor:
    def __init__(self, library, config, spec):
        self.library, self.config, self.spec = library, config, spec
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        library.trade_path.argtypes = (
            [ptr]
            + [ctypes.c_int] * 5
            + [ctypes.c_double] * 6
            + [ctypes.c_int]
            + [ctypes.c_double] * 2
            + [ptr] * 3
        )
        library.trade_path.restype = None

    def simulate(self, event, quotes, side, variant, stress=1, opposite_time=None):
        if variant not in self.config["variants"] or side not in (-1, 1):
            raise ValueError("Unknown variant/direction")
        assert quotes.dtype == np.float64 and quotes.shape[1] == 8 and len(quotes)
        slip, fee = float(slippage(float(event.entry_mid), self.spec, stress)), commission_unit(self.spec)
        entry = float(event.entry_mid) + side * (float(event.entry_spread) * stress / 2 + slip)
        distance = (2 if variant == "fixed60_stop2" else 1) * float(event.atr)
        risk = distance + slip + fee
        assert risk > 0 and np.isfinite(risk)
        opposite = (
            -1 if opposite_time is None else int((opposite_time - event.entry_time).total_seconds() / 60)
        )
        values = np.empty(8)
        path = np.empty(len(quotes) * 4 + 1)
        trace = np.empty((len(quotes), 5))
        self.library.trade_path(
            quotes,
            len(quotes),
            event.entry_time.minute,
            side,
            int(variant == "runner_stop1"),
            opposite,
            stress,
            entry,
            slip,
            fee,
            distance,
            risk,
            self.config["trail_clock_minutes"],
            self.config["trail_activation_r"],
            self.config["trail_distance_r"],
            values,
            path,
            trace,
        )
        exit_price, stop, net, minutes, reason, active, updates, length = values
        path = path[: int(length)].copy()
        trail = []
        for minute, old, new, best, locked in trace[: int(updates)]:
            time = event.entry_time + pd.Timedelta(minutes=int(minute))
            trail.append(
                {
                    "observed_close_time": time,
                    "effective_open_time": time,
                    "old_stop": old,
                    "new_stop": new,
                    "best_close_r": best,
                    "locked_r": locked,
                }
            )
        return {
            "entry_time": event.entry_time,
            "exit_time": event.entry_time + pd.Timedelta(minutes=int(minutes)),
            "day": event.day,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_price,
            "atr": float(event.atr),
            "initial_stop": entry - side * distance,
            "final_stop": stop,
            "stop_distance": distance,
            "risk_points": risk,
            "exit_slippage": slip,
            "commission_points": fee,
            "net_points": net,
            "net_r": net / risk,
            "nominal_stop_r": net / distance,
            "net_atr": net / float(event.atr),
            "mfe_r": max(0.0, float(path.max())) / risk,
            "mae_r": max(0.0, -float(path.min())) / risk,
            "exit_reason": REASONS[int(reason)],
            "stress": stress,
            "variant": variant,
            "trail_activated": bool(active),
            "trail_updates": int(updates),
            "holding_minutes": float(minutes),
            "opposite_time": opposite_time,
            "_path": path,
            "_trace": trail,
        }


def path_summary(trade, risks):
    r = np.asarray(trade["_path"]) / trade["risk_points"]
    # Do not silently approximate insolvency. Production must pass this check.
    if np.min(r) * max(risks) <= -1:
        raise ValueError("Cache compression requires solvent individual paths; use full account simulator")
    peaks = np.maximum.accumulate(np.r_[0.0, r])[1:]
    return np.r_[
        trade["net_r"],
        min(0.0, r.min()),
        max(0.0, r.max()),
        [np.max(risk * (peaks - r) / (1 + risk * peaks)) for risk in risks],
    ]


def compressed_accounts(summaries, risk, risk_index):
    """summaries [draw, chronological trade, field]; equivalent to full OHLC path."""
    r = summaries[:, :, 0]
    growth = np.cumprod(1 + risk * r, axis=1)
    before = np.c_[np.ones(len(r)), growth[:, :-1]]
    highs = before * (1 + risk * summaries[:, :, 2])
    lows = before * (1 + risk * summaries[:, :, 1])
    previous_peak = np.maximum.accumulate(np.c_[np.ones(len(r)), highs], axis=1)[:, :-1]
    cross_trade = (previous_peak - lows) / previous_peak
    local_dd = summaries[:, :, 3 + risk_index]
    dd = np.maximum(cross_trade, local_dd).max(axis=1) * 100
    return (growth[:, -1] - 1) * 100, dd
