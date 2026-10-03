import numpy as np
import pandas as pd
from .data import write_json
from .runner import account_scenario, complete_session_path, simulate
from .validation import audit_trade
from .execution import QUOTE_COLUMNS, compressed_accounts, path_summary
MONTHS = pd.period_range('2014-01', '2020-12', freq='M').astype(str).tolist()

def compare_execution(reference, actual):
    assert reference.keys() == actual.keys()
    for key in reference:
        if key == "_path":
            np.testing.assert_allclose(reference[key], actual[key], rtol=0, atol=1e-11)
        elif isinstance(reference[key], (float, np.floating)):
            assert np.isclose(reference[key], actual[key], rtol=0, atol=1e-11), key
        else:
            assert reference[key] == actual[key], key


def eligible_paths(events, m1, closes):
    parts, valid, excluded = {}, [], []
    for event in events.itertuples():
        part = complete_session_path(event, m1, closes[event.day])
        if part is None:
            excluded.append(event.Index)
        else:
            parts[event.Index] = part
            valid.append(event.Index)
    return parts, np.array(valid, dtype=int), excluded


def random_plans(events, selected, valid, replicates, seed):
    """Matched annual side/clock counts; rejection depends on availability only."""
    rng = np.random.default_rng(seed)
    n = len(selected)
    direction = np.empty((replicates, n), dtype=np.int32)
    entry = np.empty_like(direction)
    audit = []
    # key encodes event and direction. This preserves chronology when sorted.
    for year in range(2014, 2021):
        sel = selected[selected.test_year == year]
        loc = np.flatnonzero(selected.test_year.to_numpy() == year)
        universe = events[events.test_year == year]
        days = np.array(sorted(universe.day.unique()))
        lookup = {
            (r.day, r.clock): r.Index for r in universe.loc[universe.index.intersection(valid)].itertuples()
        }
        original_sides = sel.side.to_numpy()
        clocks = sel.clock.to_numpy()
        attempts = 0
        for draw in range(replicates):
            sides = rng.permutation(original_sides)
            direction[draw, loc] = sel.index.to_numpy() * 2 + (sides == 1)
            while True:
                attempts += 1
                if attempts > replicates * 10000:
                    raise RuntimeError("Random calendar matching has too few feasible assignments")
                dates = rng.choice(days, len(sel), replace=False)
                permutation = rng.permutation(len(sel))
                indices = [lookup.get((day, clock), -1) for day, clock in zip(dates, clocks[permutation])]
                if min(indices, default=0) >= 0:
                    keys = np.array(indices) * 2 + (original_sides[permutation] == 1)
                    entry[draw, loc] = np.sort(keys)
                    break
        audit.append(
            {
                "year": year,
                "trades": len(sel),
                "longs": int((original_sides == 1).sum()),
                "shorts": int((original_sides == -1).sum()),
                "eligible_days": len(days),
                "assignment_attempts": attempts,
                "accepted_assignments": replicates,
                "direction_control_informative": len(np.unique(original_sides)) > 1,
            }
        )
    assert np.all(np.diff(entry // 2, axis=1) > 0)
    for plan in [direction, entry]:
        for row in plan:
            chosen = events.iloc[row // 2]
            assert not chosen.day.duplicated().any()
            assert np.array_equal(chosen.test_year.to_numpy(), selected.test_year.to_numpy())
    return {"random_direction": direction, "random_entry": entry}, audit


def controls(events, selected, valid, parts, executor, execution, parent, output, frames):
    replicates = parent["random_replicates"]
    plans, sampling_audit = random_plans(events, selected, valid, replicates, parent["random_seed"])
    np.savez_compressed(output / "random_plans.npz", **plans)
    write_json(output / "random_sampling_audit.json", sampling_audit)
    risks = execution["risks"]
    keys = np.unique(np.concatenate([p.ravel() for p in plans.values()]))
    # Include the candidate diagnostic in the same deterministic cache.
    candidate_keys = selected.index.to_numpy() * 2 + (selected.side.to_numpy() == 1)
    keys = np.union1d(keys, candidate_keys)
    draws, comparisons, distributions = [], [], []
    reference_checks = 0
    for stress in execution["stress_multipliers"]:
        for variant in execution["variants"]:
            print(f"RANDOM CACHE {variant} cost={stress} unique entry/directions={len(keys)}", flush=True)
            cache = np.full((len(events) * 2, 3 + len(risks)), np.nan)
            audit_indices = set(np.linspace(0, len(keys) - 1, 42).astype(int))
            for i, key in enumerate(keys):
                event = events.iloc[key // 2]
                part = parts[key // 2]
                side = 1 if key % 2 else -1
                trade = executor.simulate(
                    event, np.ascontiguousarray(part[QUOTE_COLUMNS]), side, variant, stress
                )
                if i in audit_indices:
                    reference = simulate(event, part, side, executor.spec, variant, execution, stress)
                    compare_execution(reference, trade)
                    audit_trade(trade, part)
                    reference_checks += 1
                cache[key] = path_summary(trade, risks)
            np.save(output / f"cache_{variant}_{stress}.npy", cache)
            candidate = frames[(variant, stress, "no_opposite")]
            np.testing.assert_allclose(cache[candidate_keys, 0], candidate.net_r, rtol=0, atol=1e-11)
            observed = cache[candidate_keys][None]
            for control, plan in plans.items():
                samples = cache[plan]
                assert np.isfinite(samples).all()
                net = samples[:, :, 0].sum(axis=1)
                win = np.maximum(samples[:, :, 0], 0).sum(axis=1)
                loss = -np.minimum(samples[:, :, 0], 0).sum(axis=1)
                pf = np.divide(win, loss, out=np.full_like(win, np.nan), where=loss > 0)
                for risk_index, risk in enumerate(risks):
                    returns, dd = compressed_accounts(samples, risk, risk_index)
                    actual_return, actual_dd = compressed_accounts(observed, risk, risk_index)
                    baseline, _ = account_scenario(candidate, 1000, risk, MONTHS)
                    assert np.isclose(actual_return[0], baseline["return_pct"], atol=1e-9)
                    assert np.isclose(actual_dd[0], baseline["modeled_marked_dd_pct"], atol=1e-9)
                    label = {"control": control, "variant": variant, "stress": stress, "risk_pct": 100 * risk}
                    draws.append(
                        pd.DataFrame(
                            dict(
                                **label,
                                draw=np.arange(replicates),
                                trades=len(selected),
                                net_r=net,
                                pf=pf,
                                return_pct=returns,
                                modeled_marked_dd_pct=dd,
                            )
                        )
                    )
                    candidate_net = float(candidate.net_r.sum())
                    comparisons.append(
                        dict(
                            **label,
                            candidate_return_pct=float(actual_return[0]),
                            candidate_dd_pct=float(actual_dd[0]),
                            candidate_net_r=candidate_net,
                            p_upper_return=(1 + int((returns >= actual_return[0]).sum())) / (1 + replicates),
                            p_upper_net_r=(1 + int((net >= candidate_net).sum())) / (1 + replicates),
                            return_percentile=float((returns < actual_return[0]).mean() * 100),
                            primary=variant == "runner_stop1" and stress == 1 and risk == 0.01,
                        )
                    )
                    for capital in execution["capitals"]:
                        row = dict(**label, capital=capital, replicates=replicates)
                        for name, values in {
                            "net_r": net,
                            "pf": pf,
                            "return_pct": returns,
                            "marked_dd_pct": dd,
                            "final_balance": capital * (1 + returns / 100),
                        }.items():
                            for pct in [2.5, 50, 97.5]:
                                row[f"{name}_p{pct:g}"] = float(np.nanpercentile(values, pct))
                        distributions.append(row)
    pd.concat(draws, ignore_index=True).to_parquet(output / "random_draws.parquet", index=False)
    pd.DataFrame(comparisons).to_csv(output / "random_comparison.csv", index=False)
    pd.DataFrame(distributions).to_csv(output / "random_account_distributions.csv", index=False)
    return {
        "replicates_per_control": replicates,
        "seed": parent["random_seed"],
        "independent_cache_path_checks": reference_checks,
        "cache_entries_per_arm": len(keys),
        "summary_scenarios": len(distributions),
        "candidate_accounts_reconciled": True,
        "insolvency_absent_in_all_individual_cached_paths": True,
    }
