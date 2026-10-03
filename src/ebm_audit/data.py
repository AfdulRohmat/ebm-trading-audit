import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["open", "high", "low", "close"]
FEATURES = [
    "ret_1",
    "ret_2",
    "ret_4",
    "ret_8",
    "vol_8",
    "body",
    "range",
    "mean_distance",
    "range_position",
    "atr_fraction",
    "time_sin",
    "time_cos",
]



def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_features(m1):
    """Features depend only on completed bid bars; do not backfill gaps."""
    bid = m1.set_index("timestamp")[[f"bid_{c}" for c in FIELDS]].rename(columns=lambda c: c[4:])
    grouped = bid.resample("30min", label="right", closed="left")
    bars = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bars = bars[grouped.close.count().eq(30)].copy()
    previous = bars.close.shift()
    tr = pd.concat(
        [bars.high - bars.low, (bars.high - previous).abs(), (bars.low - previous).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    features = pd.DataFrame(index=bars.index)
    for lag in [1, 2, 4, 8]:
        features[f"ret_{lag}"] = (bars.close - bars.close.shift(lag)) / atr
    features["vol_8"] = bars.close.diff().rolling(8).std() / atr
    features["body"] = (bars.close - bars.open) / atr
    features["range"] = (bars.high - bars.low) / atr
    features["mean_distance"] = (bars.close - bars.close.rolling(8).mean()) / atr
    lo, hi = bars.low.rolling(8).min(), bars.high.rolling(8).max()
    features["range_position"] = (bars.close - lo) / (hi - lo).replace(0, np.nan)
    features["atr_fraction"] = atr / bars.close
    ny = bars.index.tz_convert("America/New_York")
    minute = ny.hour * 60 + ny.minute
    features["time_sin"] = np.sin(minute / 1440 * 2 * np.pi)
    features["time_cos"] = np.cos(minute / 1440 * 2 * np.pi)
    features["atr"] = atr
    features["day"] = ny.tz_localize(None).to_numpy(dtype="datetime64[D]").astype(str)
    features["signal_time"] = bars.index
    features["entry_time"] = bars.index
    mask = (minute >= 570) & (minute <= 870) & (ny.dayofweek < 5)
    return features.loc[mask].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def make_events(m1, horizon=60):
    events = make_features(m1)
    times = pd.DatetimeIndex(m1.timestamp)
    events["exit_time"] = events.entry_time + pd.Timedelta(minutes=horizon)
    starts = times.get_indexer(events.entry_time)
    ends = times.get_indexer(events.exit_time)
    valid = (starts >= 0) & (ends >= 0) & (ends - starts == horizon)
    # Valid endpoints + exactly horizon rows cannot hide gaps on a minute grid.
    if ((times.asi8 % (60 * 10**9)) != 0).any():
        raise ValueError("Non-M1-grid timestamps")
    events = events.loc[valid].copy()
    events["start_idx"] = starts[valid]
    events["end_idx"] = ends[valid]
    mid = (m1.bid_open.to_numpy() + m1.ask_open.to_numpy()) / 2
    events["target"] = (mid[ends[valid]] - mid[starts[valid]]) / events.atr.to_numpy()
    events["entry_mid"] = mid[starts[valid]]
    events["entry_spread"] = (m1.ask_open.to_numpy() - m1.bid_open.to_numpy())[starts[valid]]
    return events.reset_index(drop=True)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
