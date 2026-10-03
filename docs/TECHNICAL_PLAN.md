# EBM publication and matched-control audit

Contract written before calculating the new EBM controls. Historical EBM outcomes
have already been inspected; this is not a new holdout or a preregistered model search.

## Scope and immutable baseline

SPY proxy, 2014–2020, seven frozen annual EBM models, M30 features, 60-minute
ATR-normalized regression target. Preserve prior research, predictions, parameters,
entries, long/short decisions and all three execution variants. No retraining,
new thresholds, removal of shorts, or selection of a best risk after viewing results.
Retain costs 1x/2x, risks 1–5%, and capitals $500/$1000/$1500/$2000/$2500/$3000.
Do not imply Exness US500 feasibility or XAUUSD validation from SPY results.

## New controls

Use 2,000 draws per null, seed 261004. Match annual entry count, time-of-day and
long/short composition. Random direction keeps the original entry dates; random
entry moves entries to unique eligible dates within each year. Matching uses
observed strategy activity, so these are conditional diagnostics, not forward
deployable random strategies. Availability rejection must not inspect returns.

Random controls use a classifier-independent exit: remove opposite-EBM-signal
exits from BOTH candidate and controls. Report this diagnostic separately from
the unchanged primary runner, which keeps its opposite-signal exit. All three
variants and both cost arms are reported; runner/base/1% is the designated
primary control comparison. Do not cherry-pick the lowest p-value.

Compare with cash-funded, fractional, 1x SPY buy-and-hold over the same calendar
period, including synthetic transaction costs but excluding dividends and taxes.
Explain overnight exposure and leverage differences; do not claim apples-to-apples alpha.

## Monte Carlo and figures

2,000 circular moving-block bootstrap draws, five-session blocks, seed 261005,
including all no-trade days. Apply the observed primary-runner daily trade paths
to each resampled calendar, across all risk and capital settings. Report final
return and intratrade marked drawdown quantiles. This assumes historical blocks
are reusable; it does not estimate the probability of future profits or repair
selection bias. Show a historical-uncertainty fan, not a future forecast.

Publish equity, annual returns, risk/DD grid, conditional-null distribution,
Monte Carlo uncertainty, learned feature-contribution curves and calibration.
Learned feature curves are not fitted price paths or proof of forecast skill.

## Engineering and acceptance

1. Import a minimal, attributed snapshot of existing execution/feature code.
2. Verify prepared-data and prediction SHA256; reconstruct every frozen EBM
   prediction from portable models; reconcile primary trades/accounts with v3.
3. Compare accelerated paths with the independent Python implementation and
   audit stop timing, fill/PnL and compressed-account equivalence.
4. Check random plans preserve annual clock/side counts and one trade/day.
5. Generate results and figures from scripts, inspect plots and README links,
   run unit tests and deterministic report regeneration.
6. Preserve original repositories; commit only source, models and derived audit
   evidence. No raw market archive, credentials or local home-directory paths.
7. Publish only after repository visibility is resolved. If GitHub creation/auth
   is unavailable, deliver the complete local repository and state the blocker.

Status remains an evidence-based decision, not a promised profitable EA.
