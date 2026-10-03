import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ebm_audit.build import compile_executor
from ebm_audit.controls import compare_execution, random_plans
from ebm_audit.data import FEATURES, make_features
from ebm_audit.execution import (
    QUOTE_COLUMNS,
    Executor,
    compressed_accounts,
    path_summary,
)
from ebm_audit.model import contributions, predict
from ebm_audit.monte_carlo import block_indices
from ebm_audit.runner import account_scenario, simulate
from test_execution import CONFIG, SPEC, event, flat_path, set_bar

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    return Executor(
        compile_executor(tmp_path_factory.mktemp("execution")), CONFIG, SPEC
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("variant", CONFIG["variants"])
@pytest.mark.parametrize("stress", [1, 2])
def test_accelerator_and_compression(executor, side, variant, stress):
    frame = flat_path(121, spread=0.1)
    for k in range(len(frame)):
        op = 100 + side * k / 30
        set_bar(frame, k, op, op + 0.25, op - 0.25, op, spread=0.1)
    ev = event(frame)
    trade = executor.simulate(
        ev, np.ascontiguousarray(frame[QUOTE_COLUMNS]), side, variant, stress
    )
    compare_execution(simulate(ev, frame, side, SPEC, variant, CONFIG, stress), trade)
    compressed = path_summary(trade, CONFIG["risks"])[None, None, :]
    for i, risk in enumerate(CONFIG["risks"]):
        ret, dd = compressed_accounts(compressed, risk, i)
        exact, _ = account_scenario(pd.DataFrame([trade]), 1000, risk, ["2020-01"])
        assert ret[0] == pytest.approx(exact["return_pct"])
        assert dd[0] == pytest.approx(exact["modeled_marked_dd_pct"])


def test_portable_contributions_and_missing_bin():
    model = json.loads((ROOT / "models/fold1/portable.json").read_text())
    x = np.zeros((3, len(FEATURES)))
    x[1] = np.nan
    x[2] = [cuts[0] for cuts in model["cuts"]]
    terms = contributions(model, x)
    np.testing.assert_allclose(
        predict(model, x), model["intercept"] + terms.sum(axis=1), atol=1e-15
    )
    assert np.array_equal(terms[1], [scores[0] for scores in model["scores"]])
    assert np.array_equal(terms[2], [scores[2] for scores in model["scores"]])
    with pytest.raises(ValueError):
        predict(model, np.zeros((1, 2)))


def test_block_sampling_keeps_contiguous_sessions_and_is_deterministic():
    idx = block_indices(17, 100, 5, 77)
    np.testing.assert_array_equal(idx, block_indices(17, 100, 5, 77))
    assert idx.shape == (100, 17)
    for start in (0, 5, 10):
        assert np.all(np.diff(idx[:, start : start + 5], axis=1) % 17 == 1)


def test_random_controls_preserve_all_matching_constraints():
    rows = []
    for year in range(2014, 2021):
        for day in range(1, 11):
            for clock in (600, 630):
                rows.append(
                    {"test_year": year, "day": f"{year}-01-{day:02}", "clock": clock}
                )
    events = pd.DataFrame(rows)
    selected = events[events.index % 20 < 4].drop_duplicates("day").copy()
    selected["side"] = np.tile([1, -1], 7)
    plans, _ = random_plans(events, selected, events.index.to_numpy(), 30, 79)
    for name, matrix in plans.items():
        for row in matrix:
            sample = events.iloc[row // 2].copy()
            sample["side"] = np.where(row % 2, 1, -1)
            assert not sample.day.duplicated().any()
            for year in range(2014, 2021):
                original = selected[selected.test_year == year]
                actual = sample[sample.test_year == year]
                assert sorted(original.side) == sorted(actual.side)
                assert sorted(original.clock) == sorted(actual.clock)
                if name == "random_entry":
                    assert sorted(zip(original.clock, original.side)) == sorted(
                        zip(actual.clock, actual.side)
                    )
            if name == "random_direction":
                assert sample.day.tolist() == selected.day.tolist()


def test_flat_days_do_not_change_final_return_or_drawdown():
    cache = np.zeros((1, 3, 8))
    cache[0, 1, :3] = [-1, -1, 0]
    cache[0, 1, 3:] = CONFIG["risks"]
    for i, risk in enumerate(CONFIG["risks"]):
        ret, dd = compressed_accounts(cache, risk, i)
        assert ret[0] == pytest.approx(-100 * risk)
        assert dd[0] == pytest.approx(100 * risk)


def test_features_cannot_read_future_prices():
    days = []
    for day in ("2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"):
        days.append(flat_path(390, start=f"{day} 14:30Z", spread=0.1))
    frame = pd.concat(days, ignore_index=True)
    for field in QUOTE_COLUMNS:
        frame[field] += np.sin(np.arange(len(frame)) / 50)
    cutoff = 3 * 390
    prefix = make_features(frame.iloc[:cutoff])
    full = make_features(frame)
    pd.testing.assert_frame_equal(prefix, full.iloc[: len(prefix)], check_exact=True)
    frame.loc[cutoff:, QUOTE_COLUMNS] *= 10
    changed = make_features(frame)
    pd.testing.assert_frame_equal(prefix, changed.iloc[: len(prefix)], check_exact=True)


def test_committed_evidence_reconciles():
    from ebm_audit.data import checksum

    hashes = json.loads((ROOT / "results/checksums.json").read_text())
    for name, expected in hashes.items():
        assert checksum(ROOT / "results" / name) == expected
    accounts = pd.read_csv(ROOT / "results/accounts.csv")
    actual = accounts[
        (accounts.variant == "runner_stop1")
        & (accounts["mode"] == "opposite")
        & (accounts.capital == 1000)
        & (accounts.risk_pct == 1)
        & (accounts.stress == 1)
    ].iloc[0]
    assert actual.final_balance == pytest.approx(1334.956332, abs=1e-6)
    assert actual.modeled_marked_dd_pct == pytest.approx(20.966125, abs=1e-6)
    assert len(accounts) == 240
