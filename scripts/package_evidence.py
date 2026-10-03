"""Copy an allowlisted set of derived results, never raw bars or input predictions."""

import argparse
import json
import shutil
from pathlib import Path

from ebm_audit.data import checksum

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "accounts.csv",
    "summary.csv",
    "trades.csv",
    "account_ledger_1000.csv",
    "audit.json",
    "buy_hold_accounts.csv",
    "buy_hold_daily_1000.csv",
    "shape_grid.csv",
    "calibration.csv",
    "forecast_metrics.csv",
    "importance.csv",
    "historical_summary.csv",
    "historical_annual.csv",
    "historical_monthly.csv",
    "random_comparison.csv",
    "random_account_distributions.csv",
    "random_draws.parquet",
    "random_sampling_audit.json",
]
FILES += [
    f"bootstrap_cost{s}/{f}.csv"
    for s in [1, 2]
    for f in ["bootstrap_accounts", "bootstrap_bands"]
]


def main(source, destination):
    if destination.exists():
        raise FileExistsError("Evidence is immutable; choose a fresh destination")
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    hashes = {name: checksum(destination / name) for name in FILES}
    (destination / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    main(args.source, args.destination)
