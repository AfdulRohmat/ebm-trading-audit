"""Historical moving-block uncertainty; never a future-profit forecast."""

import numpy as np
import pandas as pd

from .execution import compressed_accounts


def block_indices(sessions, draws, block, seed):
    if min(sessions, draws, block) < 1:
        raise ValueError("Positive dimensions required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, sessions, size=(draws, (sessions + block - 1) // block))
    return ((starts[:, :, None] + np.arange(block)) % sessions).reshape(draws, -1)[
        :, :sessions
    ]


def run(daily, days, config, output):
    idx = block_indices(
        len(days),
        config["bootstrap_draws"],
        config["bootstrap_block_sessions"],
        config["bootstrap_seed"],
    )
    samples = daily[idx]
    rows, bands = [], []
    for k, risk in enumerate(config["risks"]):
        returns, dd = compressed_accounts(samples, risk, k)
        paths = np.cumprod(1 + risk * samples[:, :, 0], axis=1)
        for capital in config["capitals"]:
            row = {"risk_pct": 100 * risk, "capital": capital, "draws": len(idx)}
            for name, values in {
                "return_pct": returns,
                "marked_dd_pct": dd,
                "final_balance": capital * (1 + returns / 100),
            }.items():
                for q in (2.5, 50, 97.5):
                    row[f"{name}_p{q:g}"] = float(np.percentile(values, q))
            row["resample_positive_fraction"] = float((returns > 0).mean())
            rows.append(row)
        quantiles = np.percentile(paths, [2.5, 50, 97.5], axis=0)
        bands.append(
            pd.DataFrame(
                {
                    "day": days,
                    "risk_pct": 100 * risk,
                    "p2.5": quantiles[0],
                    "p50": quantiles[1],
                    "p97.5": quantiles[2],
                }
            )
        )
    pd.DataFrame(rows).to_csv(output / "bootstrap_accounts.csv", index=False)
    pd.concat(bands).to_csv(output / "bootstrap_bands.csv", index=False)
    return {
        "draws": len(idx),
        "sessions": len(days),
        "block_sessions": config["bootstrap_block_sessions"],
        "seed": config["bootstrap_seed"],
        "includes_flat_days": True,
        "future_forecast": False,
    }
