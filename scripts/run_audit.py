"""Replay frozen EBM, matched random controls and historical uncertainty.

Requires the private research archive, not included in this publicable repository.
Writes only to a fresh output directory; never edits the source archive.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from ebm_audit.build import compile_executor
from ebm_audit.controls import compare_execution, controls, eligible_paths
from ebm_audit.data import FEATURES, checksum, write_json
from ebm_audit.execution import QUOTE_COLUMNS, Executor, path_summary
from ebm_audit.model import predict
from ebm_audit.monte_carlo import run as bootstrap_run
from ebm_audit.runner import account_scenario, simulate
from ebm_audit.trading import directions, metrics
from ebm_audit.validation import audit_trade

ROOT = Path(__file__).resolve().parents[1]
MONTHS = pd.period_range("2014-01", "2020-12", freq="M").astype(str).tolist()


def run(source, output):
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    cfg = json.loads((ROOT / "config/audit.json").read_text())
    prediction = source / "artifacts/spy_v2/predictions.parquet"
    prepared = source / "data/processed/spy_v2_m1.parquet"
    manifest = json.loads((source / "artifacts/spy_v2/run_manifest.json").read_text())
    data_audit = json.loads((source / "artifacts/spy_v2/data_audit.json").read_text())
    assert checksum(prediction) == manifest["prediction_sha256"]
    assert checksum(prepared) == data_audit["prepared_sha256"]
    # Include user-owned untracked deep-dive evidence in the preservation audit.
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=source
        )
        .decode()
        .split("\0")
    )
    protected = {n: checksum(source / n) for n in set(names) if n}
    protected.update(
        {str(p.relative_to(source)): checksum(p) for p in [prediction, prepared]}
    )
    events = (
        pd.read_parquet(prediction).sort_values("entry_time").reset_index(drop=True)
    )
    m1 = pd.read_parquet(prepared)
    parity = 0.0
    for fold, group in events.groupby("fold"):
        model = json.loads((ROOT / f"models/fold{fold}/portable.json").read_text())
        error = float(
            np.max(np.abs(predict(model, group[FEATURES]) - group.ebm.to_numpy()))
        )
        parity = max(parity, error)
    assert parity < 1e-12
    ny = events.entry_time.dt.tz_convert("America/New_York")
    events["clock"] = ny.dt.hour * 60 + ny.dt.minute
    events["side"] = directions(
        events, events.ebm, cfg["source_spec"], cfg["threshold_atr"]
    )
    selected = events[events.side.ne(0)].drop_duplicates("day", keep="first")
    schedule = xcals.get_calendar("XNYS", start="2014-01-01", end="2021-01-01").schedule
    closes = {str(day.date()): row.close for day, row in schedule.iterrows()}
    days = sorted(events.day.unique())
    print(f"ELIGIBILITY {len(events)} events; {len(selected)} EBM entries", flush=True)
    parts, valid, invalid = eligible_paths(events, m1, closes)
    assert len(valid) == len(events) and not invalid
    assert len(selected) == 362 and len(days) == 1763
    output.mkdir(parents=True)
    originals = pd.read_csv(source / "artifacts/spy_v3_runner/trades.csv")
    original_accounts = pd.read_csv(source / "artifacts/spy_v3_runner/accounts.csv")
    frames, trade_tables, accounts, ledger_tables, summaries = {}, [], [], [], []
    source_hashes = {
        "prepared_m1": checksum(prepared),
        "predictions": checksum(prediction),
    }
    with tempfile.TemporaryDirectory(prefix="ebm-audit-") as temporary:
        executor = Executor(compile_executor(temporary), cfg, cfg["source_spec"])
        for stress in cfg["stress_multipliers"]:
            for variant in cfg["variants"]:
                for mode in (
                    ["opposite", "no_opposite"]
                    if variant == "runner_stop1"
                    else ["opposite"]
                ):
                    rows = []
                    for event in selected.itertuples():
                        later = events[
                            (events.day == event.day)
                            & (events.entry_time > event.entry_time)
                            & (events.side == -event.side)
                        ]
                        opposite = (
                            None
                            if mode == "no_opposite" or later.empty
                            else later.entry_time.iloc[0]
                        )
                        part = parts[event.Index]
                        trade = executor.simulate(
                            event,
                            np.ascontiguousarray(part[QUOTE_COLUMNS]),
                            int(event.side),
                            variant,
                            stress,
                            opposite,
                        )
                        compare_execution(
                            simulate(
                                event,
                                part,
                                int(event.side),
                                cfg["source_spec"],
                                variant,
                                cfg,
                                stress,
                                opposite,
                            ),
                            trade,
                        )
                        audit_trade(trade, part)
                        trade.update(
                            test_year=int(event.test_year), fold=int(event.fold)
                        )
                        rows.append(trade)
                    frame = pd.DataFrame(rows)
                    frames[(variant, stress, mode)] = frame
                    if variant != "runner_stop1":
                        frames[(variant, stress, "no_opposite")] = frame
                    label = {"variant": variant, "stress": stress, "mode": mode}
                    trade_tables.append(
                        frame.drop(columns=["_path", "_trace"]).assign(**label)
                    )
                    summary = metrics(frame)
                    summary["stop_rate"] = float(
                        frame.exit_reason.isin(["initial_stop", "trailing_stop"]).mean()
                    )
                    summaries.append(dict(**label, **summary))
                    if mode == "opposite":
                        old = originals[
                            (originals.model == "ebm")
                            & (originals.variant == variant)
                            & (originals.stress == stress)
                        ]
                        np.testing.assert_allclose(
                            frame.net_r, old.net_r, rtol=0, atol=1e-11
                        )
                        assert (
                            pd.to_datetime(old.entry_time, utc=True).tolist()
                            == frame.entry_time.tolist()
                        )
                        assert (
                            pd.to_datetime(old.exit_time, utc=True).tolist()
                            == frame.exit_time.tolist()
                        )
                        assert old.exit_reason.tolist() == frame.exit_reason.tolist()
                    for capital in cfg["capitals"]:
                        for risk in cfg["risks"]:
                            account, ledger = account_scenario(
                                frame, capital, risk, MONTHS
                            )
                            accounts.append(dict(**label, **account))
                            if mode == "opposite":
                                old = original_accounts[
                                    (original_accounts.model == "ebm")
                                    & (original_accounts.variant == variant)
                                    & (original_accounts.stress == stress)
                                    & (original_accounts.capital == capital)
                                    & (original_accounts.risk_pct == 100 * risk)
                                ].iloc[0]
                                for name in [
                                    "final_balance",
                                    "return_pct",
                                    "modeled_marked_dd_pct",
                                ]:
                                    assert np.isclose(
                                        account[name], old[name], atol=1e-8, rtol=0
                                    ), name
                            if capital == 1000:
                                ledger_tables.append(
                                    ledger.assign(**label, risk_pct=100 * risk)
                                )
                    print(f"CANDIDATE {label}: {frame.net_r.sum():.5f}R", flush=True)
        pd.concat(trade_tables).to_csv(output / "trades.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(accounts).to_csv(output / "accounts.csv", index=False)
        pd.concat(ledger_tables).to_csv(output / "account_ledger_1000.csv", index=False)
        null_audit = controls(
            events, selected, valid, parts, executor, cfg, cfg, output, frames
        )
        mc_audits = []
        for stress in cfg["stress_multipliers"]:
            frame = frames[("runner_stop1", stress, "opposite")]
            daily = np.zeros((len(days), 3 + len(cfg["risks"])))
            positions = {d: i for i, d in enumerate(days)}
            for trade in frame.to_dict("records"):
                daily[positions[trade["day"]]] = path_summary(trade, cfg["risks"])
            mc_dir = output / f"bootstrap_cost{stress}"
            mc_dir.mkdir()
            mc_audits.append(bootstrap_run(daily, days, cfg, mc_dir))
        for name in ["buy_hold_accounts.csv", "buy_hold_daily_1000.csv"]:
            src = source / "artifacts/spy_v5_continuous" / name
            shutil.copyfile(src, output / name)
            source_hashes[name] = checksum(src)
        for name in [
            "shape_grid.csv",
            "calibration.csv",
            "forecast_metrics.csv",
            "importance.csv",
        ]:
            src = source / "artifacts/ebm_deep_dive" / name
            shutil.copyfile(src, output / name)
            source_hashes[name] = checksum(src)
        for name in ["summary.csv", "annual.csv", "monthly.csv"]:
            src = source / "artifacts/spy_v3_runner" / name
            frame = pd.read_csv(src)
            frame = frame[frame.model == "ebm"].copy()
            if name in ["annual.csv", "monthly.csv"]:
                # The legacy generic metrics helper counted only literal "stop".
                # Preserve performance values but repair this descriptive rate.
                for idx, row in frame.iterrows():
                    trades = frames[(row.variant, row.stress, "opposite")]
                    if name == "annual.csv":
                        trades = trades[trades.test_year == row.year]
                    else:
                        trades = trades[trades.day.str[:7] == row.month]
                    frame.loc[idx, "stop_rate"] = (
                        float(
                            trades.exit_reason.isin(
                                ["initial_stop", "trailing_stop"]
                            ).mean()
                        )
                        if len(trades)
                        else None
                    )
            frame.to_csv(output / f"historical_{name}", index=False)
            source_hashes[f"historical_{name}"] = checksum(src)
    for name, expected in protected.items():
        assert checksum(source / name) == expected, f"Source changed: {name}"
    write_json(
        output / "audit.json",
        {
            "prediction_max_abs_error": parity,
            "events": len(events),
            "selected": len(selected),
            "sessions": len(days),
            "all_candidate_paths_python_cpp_parity": True,
            "all_primary_trades_and_accounts_match_v3": True,
            "preserved_source_files": len(protected),
            "source_hashes": source_hashes,
            "random_controls": null_audit,
            "bootstrap": mc_audits,
            "retrained": False,
            "pristine_oos": False,
            "broker_validated": False,
            "contract_sha256": checksum(ROOT / "config/audit.json"),
        },
    )
    print("AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
