import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ebm_audit.data import ROOT
from ebm_audit.validation import audit_trade
from ebm_audit.runner import account_scenario, complete_session_path, simulate

CONFIG = json.loads((ROOT / "config/audit.json").read_text())
SPEC = {"contract": 1, "commission_side_lot": 0, "slippage_point_side": 0.0}


def flat_path(n=121, start="2020-01-02 15:00Z", spread=0.0):
    times = pd.date_range(start, periods=n, freq="min")
    frame = pd.DataFrame({"timestamp": times})
    for field in ["open", "high", "low", "close"]:
        frame[f"bid_{field}"] = 100.0 - spread / 2
        frame[f"ask_{field}"] = 100.0 + spread / 2
    return frame


def event(frame, atr=1.0):
    return SimpleNamespace(
        start_idx=0,
        entry_time=frame.timestamp.iloc[0],
        day="2020-01-02",
        atr=atr,
        entry_mid=100.0,
        entry_spread=frame.ask_open.iloc[0] - frame.bid_open.iloc[0],
    )


def set_bar(frame, index, o, h, l, c, spread=0.0):
    for side, offset in [("bid", -spread / 2), ("ask", spread / 2)]:
        for field, value in zip(["open", "high", "low", "close"], [o, h, l, c]):
            frame.loc[index, f"{side}_{field}"] = value + offset


@pytest.mark.parametrize("side", [-1, 1])
def test_flat_costs_are_not_counted_twice(side):
    frame = flat_path(spread=0.1)
    spec = dict(SPEC, slippage_point_side=0.05, commission_side_lot=0.035)
    row = simulate(event(frame), frame, side, spec, "runner_stop1", CONFIG)
    assert row["net_points"] == pytest.approx(-0.27)
    assert row["risk_points"] == pytest.approx(1.12)
    assert row["exit_reason"] == "session_close"
    audit_trade(row, frame)


@pytest.mark.parametrize("side", [-1, 1])
def test_initial_stop_is_exact_budget_without_gap(side):
    frame = flat_path(spread=0.1)
    set_bar(frame, 1, 100, 102, 98, 100, spread=0.1)
    spec = dict(SPEC, slippage_point_side=0.05, commission_side_lot=0.035)
    row = simulate(event(frame), frame, side, spec, "runner_stop1", CONFIG)
    assert row["net_r"] == pytest.approx(-1.0)
    assert row["exit_reason"] == "initial_stop"
    assert row["exit_time"] == frame.timestamp.iloc[1]
    audit_trade(row, frame)


@pytest.mark.parametrize("side", [-1, 1])
def test_gap_can_exceed_allocated_risk(side):
    frame = flat_path()
    set_bar(frame, 1, 100 - 3 * side, 100 - 3 * side, 100 - 3 * side, 100 - 3 * side)
    row = simulate(event(frame), frame, side, SPEC, "runner_stop1", CONFIG)
    assert row["net_r"] == pytest.approx(-3)
    audit_trade(row, frame)


@pytest.mark.parametrize("side", [-1, 1])
def test_trail_uses_completed_close_and_is_not_retroactive(side):
    frame = flat_path()
    if side == 1:
        set_bar(frame, 29, 100, 102.5, 99.5, 102)
        set_bar(frame, 30, 102, 103, 100.5, 102)
    else:
        set_bar(frame, 29, 100, 100.5, 97.5, 98)
        set_bar(frame, 30, 98, 99.5, 97, 98)
    row = simulate(event(frame), frame, side, SPEC, "runner_stop1", CONFIG)
    assert row["exit_time"] == frame.timestamp.iloc[30]
    assert row["net_r"] == pytest.approx(1)
    assert row["exit_reason"] == "trailing_stop"
    assert row["_trace"][0]["effective_open_time"] == frame.timestamp.iloc[30]
    audit_trade(row, frame)


def test_intrabar_high_does_not_activate_trail():
    frame = flat_path()
    set_bar(frame, 29, 100, 105, 99.5, 100.5)
    row = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    assert not row["trail_activated"]
    assert row["exit_reason"] == "session_close"


def test_trail_never_loosens_and_winner_is_not_capped():
    frame = flat_path(n=91)
    for i in range(len(frame)):
        level = 100 if i < 29 else 102 if i < 59 else 101.5 if i < 89 else 108
        set_bar(frame, i, level, level, level, level)
    row = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    assert row["net_r"] == pytest.approx(8)
    assert [x["new_stop"] for x in row["_trace"]] == [101, 107]
    assert row["exit_reason"] == "session_close"
    audit_trade(row, frame)


def test_opposite_signal_exits_at_open_not_later_low():
    frame = flat_path()
    set_bar(frame, 30, 100.5, 101, 95, 100)
    row = simulate(
        event(frame), frame, 1, SPEC, "runner_stop1", CONFIG, opposite_time=frame.timestamp.iloc[30]
    )
    assert row["exit_reason"] == "opposite"
    assert row["net_r"] == pytest.approx(0.5)
    audit_trade(row, frame)


def test_stop_gap_precedes_opposite_open_exit():
    frame = flat_path()
    set_bar(frame, 30, 98, 98, 98, 98)
    row = simulate(
        event(frame), frame, 1, SPEC, "runner_stop1", CONFIG, opposite_time=frame.timestamp.iloc[30]
    )
    assert row["exit_reason"] == "initial_stop"
    assert row["net_r"] == pytest.approx(-2)
    audit_trade(row, frame)


def test_future_after_exit_cannot_change_outcome():
    frame = flat_path()
    set_bar(frame, 1, 100, 100, 98, 100)
    before = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    frame.loc[2:, [c for c in frame if c != "timestamp"]] += 50
    after = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    assert before == after


def test_missing_path_is_rejected_not_forward_filled():
    frame = flat_path()
    close = frame.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
    assert complete_session_path(event(frame), frame, close) is not None
    broken = frame.drop(index=70)
    assert complete_session_path(event(frame), broken, close) is None
    with pytest.raises(ValueError, match="complete M1"):
        simulate(event(frame), broken, 1, SPEC, "runner_stop1", CONFIG)


def test_early_close_exits_on_final_quote_not_next_day():
    frame = flat_path(180, start="2020-11-27 15:00Z")
    close = pd.Timestamp("2020-11-27 18:00Z")
    assert complete_session_path(event(frame), frame, close) is not None
    row = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    assert row["exit_time"] == close
    audit_trade(row, frame)


def test_fixed_60_does_not_read_endpoint_high_low():
    frame = flat_path()
    set_bar(frame, 60, 100.5, 105, 95, 100)
    for variant in ["fixed60_stop1", "fixed60_stop2"]:
        row = simulate(event(frame), frame, 1, SPEC, variant, CONFIG)
        assert row["exit_reason"] == "horizon"
        assert row["net_points"] == pytest.approx(0.5)
        audit_trade(row, frame)


@pytest.mark.parametrize("risk", [0.01, 0.02, 0.03, 0.04, 0.05])
def test_exact_risk_sizing_compounding_and_capital_invariance(risk):
    frame = flat_path()
    set_bar(frame, 1, 100, 100, 98, 100)
    trade = simulate(event(frame), frame, 1, SPEC, "runner_stop1", CONFIG)
    rows = pd.DataFrame([trade, trade])
    returns = []
    for capital in CONFIG["capitals"]:
        result, history = account_scenario(rows, capital, risk, ["2020-01"])
        assert result["final_balance"] == pytest.approx(capital * (1 - risk) ** 2)
        assert result["max_loss_to_budget"] == pytest.approx(1)
        np.testing.assert_allclose(history.units * rows.risk_points, history.risk_budget)
        returns.append(result["return_pct"])
    np.testing.assert_allclose(returns, returns[0])


def test_insolvency_stops_account_without_negative_sizing():
    rows = pd.DataFrame(
        [
            {
                "day": "2020-01-02",
                "entry_time": pd.Timestamp("2020-01-02", tz="UTC"),
                "exit_time": pd.Timestamp("2020-01-02", tz="UTC"),
                "entry_price": 100,
                "risk_points": 1,
                "net_points": 2,
                "_path": [-30, 2],
            }
        ]
    )
    result, history = account_scenario(rows, 500, 0.05, ["2020-01"])
    assert result["insolvent"]
    assert result["min_equity"] < 0
    assert result["final_balance"] < 0
    assert len(history) == 1


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("stress", [1, 2])
def test_random_paths_match_independent_vector_audit(side, stress):
    rng = np.random.default_rng(17)
    for _ in range(20):
        frame = flat_path(181, spread=0.04)
        levels = 100 + np.r_[0, np.cumsum(rng.normal(0, 0.17, 180))]
        for i, op in enumerate(levels):
            close = op + rng.normal(0, 0.05)
            set_bar(frame, i, op, max(op, close) + 0.08, min(op, close) - 0.08, close, spread=0.04)
        for variant in CONFIG["variants"]:
            row = simulate(
                event(frame),
                frame,
                side,
                dict(SPEC, slippage_point_side=0.01),
                variant,
                CONFIG,
                stress,
                frame.timestamp.iloc[90],
            )
            audit_trade(row, frame)
