"""Regenerate every README chart using only committed, derived evidence."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#059669", "#dc2626"


def generate(source, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
        }
    )

    def save(fig, name, note):
        fig.text(0.04, 0.015, note, fontsize=8, color="#475569")
        fig.tight_layout(rect=[0, 0.045, 1, 0.95])
        fig.savefig(
            output / f"{name}.png", dpi=170, metadata={"Software": "ebm-trading-audit"}
        )
        plt.close(fig)

    ledger = pd.read_csv(source / "account_ledger_1000.csv")
    bh = pd.read_csv(source / "buy_hold_daily_1000.csv")
    days = pd.DatetimeIndex(sorted(bh.day.unique()))
    primary = ledger[
        (ledger.variant == "runner_stop1")
        & (ledger["mode"] == "opposite")
        & (ledger.risk_pct == 1)
    ]
    equity = {}
    for stress, label in [(1, "EBM runner / base cost"), (2, "EBM runner / 2x cost")]:
        a = primary[primary.stress == stress].set_index("day").balance
        a.index = pd.to_datetime(a.index)
        equity[label] = a.reindex(days).ffill().fillna(1000)
    a = bh[bh.stress == 1].set_index("day").balance
    a.index = pd.to_datetime(a.index)
    equity["Buy & hold / 1x cash SPY"] = a.reindex(days)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for (label, values), color in zip(equity.items(), [BLUE, ORANGE, GREEN]):
        axes[0].plot(days, values, label=label, color=color, lw=1.7)
        peak = np.maximum.accumulate(np.r_[1000, values])[1:]
        axes[1].plot(days, 100 * (values / peak - 1), color=color, lw=1)
    axes[0].set(
        ylabel="Account balance (USD)",
        title="Historical equity: SPY, 2014–2020, starting $1,000",
    )
    axes[0].legend(loc="upper left")
    axes[1].set(ylabel="Close-to-close DD (%)")
    save(
        fig,
        "equity",
        "EBM: 1% planned stop risk, intraday only. B&H: 1x cash, overnight, no dividends.\nCurves use session-end balance; headline marked drawdown also includes intratrade OHLC excursions.",
    )

    annual = pd.read_csv(source / "historical_annual.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for stress, color, offset in [(1, BLUE, -0.18), (2, ORANGE, 0.18)]:
        a = annual[(annual.variant == "runner_stop1") & (annual.stress == stress)]
        axes[0].bar(
            a.year + offset, a.net_r, 0.36, label=f"{stress}x cost", color=color
        )
    a = annual[(annual.variant == "runner_stop1") & (annual.stress == 1)]
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(title="Annual net R", ylabel="Sum of trade R")
    axes[0].legend()
    axes[1].bar(a.year, a.trades, color=BLUE)
    axes[1].set(title="Activity is not stable", ylabel="Trades per year")
    save(
        fig,
        "annual",
        "Unchanged EBM primary runner. Seven historical test years; not a pristine researcher holdout.",
    )

    accounts = pd.read_csv(source / "accounts.csv")
    a = accounts[
        (accounts.variant == "runner_stop1")
        & (accounts["mode"] == "opposite")
        & (accounts.capital == 1000)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for stress, color in [(1, BLUE), (2, ORANGE)]:
        b = a[a.stress == stress]
        axes[0].plot(
            b.risk_pct, b.return_pct, "o-", color=color, label=f"{stress}x cost"
        )
        axes[1].plot(b.risk_pct, b.modeled_marked_dd_pct, "o-", color=color)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(
        title="Seven-year total return",
        ylabel="Return (%)",
        xlabel="Planned risk per trade (%)",
        xticks=range(1, 6),
    )
    axes[0].legend()
    axes[1].set(
        title="Intratrade marked drawdown",
        ylabel="Maximum drawdown (%)",
        xlabel="Planned risk per trade (%)",
        xticks=range(1, 6),
    )
    save(
        fig,
        "risk",
        "Fractional sizing; no broker lot/margin constraints. Raising risk does not improve the underlying edge.\nPercentage results are identical for $500–$3,000 only because these are idealized account simulations.",
    )

    draws = pd.read_parquet(source / "random_draws.parquet")
    comparisons = pd.read_csv(source / "random_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, control, title in zip(
        axes,
        ["random_entry", "random_direction"],
        ["Random entry dates", "Shuffled direction, fixed dates"],
    ):
        mask = (
            (draws.variant == "runner_stop1")
            & (draws.stress == 1)
            & (draws.risk_pct == 1)
            & (draws.control == control)
        )
        c = comparisons[
            (comparisons.variant == "runner_stop1")
            & (comparisons.stress == 1)
            & (comparisons.risk_pct == 1)
            & (comparisons.control == control)
        ].iloc[0]
        ax.hist(draws.loc[mask, "return_pct"], bins=45, color=BLUE, alpha=0.8)
        ax.axvline(
            c.candidate_return_pct,
            color=RED,
            lw=2,
            label=f"EBM diagnostic: {c.candidate_return_pct:.1f}%",
        )
        ax.set(
            title=f"{title}\nconditional upper-tail p = {c.p_upper_return:.4f}",
            xlabel="Seven-year total return (%)",
            ylabel="Draws",
        )
        ax.legend(fontsize=9)
    save(
        fig,
        "random_controls",
        "2,000 draws/null; matched annual activity. Both sides use NO opposite-EBM-signal exit.\nBase costs, 1% risk. Conditional diagnostics: no correction for the earlier strategy search or multiple comparisons.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, stress, color in zip(axes, [1, 2], [BLUE, ORANGE]):
        bands = pd.read_csv(source / f"bootstrap_cost{stress}/bootstrap_bands.csv")
        b = bands[bands.risk_pct == 1]
        x = np.arange(1, len(b) + 1)
        ax.fill_between(
            x,
            1000 * b["p2.5"],
            1000 * b["p97.5"],
            color=color,
            alpha=0.18,
            label="95% pointwise resampling band",
        )
        ax.plot(x, 1000 * b.p50, color=color, lw=1.5, label="Resampled median")
        label = "EBM runner / base cost" if stress == 1 else "EBM runner / 2x cost"
        ax.plot(x, equity[label], color="#111827", lw=1, label="Actual historical path")
        ax.axhline(1000, color="gray", ls="--", lw=0.8)
        ax.set(
            title=f"Historical block bootstrap / {stress}x cost",
            xlabel="Resampled session (not a future date)",
            ylabel="Balance (USD), 1% risk",
        )
        ax.legend(fontsize=7.5, loc="upper left")
    save(
        fig,
        "monte_carlo",
        "2,000 circular five-session block draws, including flat days. Uses the primary runner WITH opposite exits.\nPointwise bands are not simultaneous confidence bounds and are NOT probabilities of future account performance.",
    )

    shape = pd.read_csv(source / "shape_grid.csv")
    features = list(shape.feature.unique())
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    for ax, feature in zip(axes.flat, features):
        for year, group in shape[shape.feature == feature].groupby("test_year"):
            ax.step(
                group.value, group.score, where="post", lw=1, alpha=0.8, label=str(year)
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(title=feature, ylabel="Contribution (ATR)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False)
    save(
        fig,
        "feature_curves",
        "Seven frozen models; pooled test-distribution 5th–95th percentile grid for retrospective explanation only.\nLearned feature contributions, NOT fitted price curves; correlated inputs and overlapping training windows limit interpretation.",
    )

    cal = pd.read_csv(source / "calibration.csv")
    cal = cal[cal.scope == "all_test"]
    importance = pd.read_csv(source / "importance.csv")
    imp = importance[importance.scope == "all_test"].sort_values(
        "mean_abs_contribution"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(cal.decile, cal.prediction, "o-", color=BLUE, label="Mean prediction")
    axes[0].plot(
        cal.decile, cal.actual, "o-", color=ORANGE, label="Mean observed 60m return"
    )
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].set(
        title="Forecast calibration is weak",
        xlabel="Prediction decile (pooled test events)",
        ylabel="Return / entry ATR",
    )
    axes[0].legend(fontsize=8)
    axes[1].barh(imp.feature, imp.mean_abs_contribution, color=BLUE)
    axes[1].set(
        title="What moves the prediction?", xlabel="Mean absolute contribution (ATR)"
    )
    save(
        fig,
        "forecast",
        "17,534 chronological test events. Correlation 0.0074; MSE 0.229% worse than the training-mean baseline.\nImportance describes model behavior, not causal drivers or demonstrated trading edge. Calibration lines have no confidence intervals.",
    )
    print(f"Generated 7 charts in {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    a = p.parse_args()
    generate(a.source, a.output)
