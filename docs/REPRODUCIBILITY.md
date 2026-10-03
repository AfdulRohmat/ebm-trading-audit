# Reproducibility and boundaries

## Three different tasks

1. **Rebuild charts:** fully supported from committed `results/` files. No market
   archive, model refit, broker access or GitHub authentication is required.
2. **Replay the market audit:** requires private, hash-matched prepared M1 data,
   frozen predictions and the original historical evidence. These are not all
   shipped here. `scripts/run_audit.py` fails if the source hashes differ, output
   already exists, or candidate trades/accounts fail reconciliation.
3. **Train a new model:** `model.fit` records the exact family/grid recipe, but
   retraining is not the publication experiment. Acquisition, normalization,
   folds and new holdout governance must be supplied separately. Retraining with
   today's defaults or another feed is not an exact replay of these results.

## Full replay with the original private archive

Activate the environment described in the README. Required archive layout:

```text
intraday-ml-return-research/
  .git/
  data/processed/spy_v2_m1.parquet
  artifacts/spy_v2/predictions.parquet
  artifacts/spy_v2/{run_manifest,data_audit}.json
  artifacts/spy_v3_runner/{trades,accounts,summary,annual,monthly}.csv
  artifacts/spy_v5_continuous/{buy_hold_accounts,buy_hold_daily_1000}.csv
  artifacts/ebm_deep_dive/{shape_grid,calibration,forecast_metrics,importance}.csv
```

The script uses the original archive's git index only to hash and preserve its
files. It does not modify that repository. Do not run it concurrently with an
editor changing those source files. `ebm_deep_dive` was a previously untracked,
user-owned analysis, explicitly included in this preservation check.

```bash
python scripts/run_audit.py \
  --source ../intraday-ml-return-research \
  --output runs/replay-02
python scripts/package_evidence.py runs/replay-02 --destination runs/evidence-02
python scripts/generate_figures.py --source runs/evidence-02 --output runs/figures-02
```

Use a new output directory for every run. The two private input hashes are
recorded in [audit.json](../results/audit.json); the Kaggle archive originally
used SHA256 `8bd52867b248359db437f2fd809ac1faafc3f69449679aae9d499423487ccd5e`.
Different Kaggle versions are not interchangeable. Timezone was inferred as
America/Denver; the raw-versus-adjusted trade-price provenance was not established.
This is an explicit data-quality limitation, not provider-verified truth.

No raw bars, full event-feature matrix or full prediction archive is committed.
Derived trade results contain timestamps and fills for auditability. Public
redistribution of the underlying archive has not been assumed. Thus a fresh
public clone cannot independently recreate the market results without obtaining
and preparing the matching source data. This limitation is intentional and
must not be described as fully self-contained end-to-end reproducibility.

## Which values are newly calculated and which are imported?

- Newly replayed: all EBM trade paths, three exits, both costs, opposite/no-opposite
  diagnostic, all accounts, 2,000 matched random draws per null and bootstrap.
- Imported and hash-recorded: earlier B&H benchmark, feature/forecast diagnostics,
  annual/monthly tables, historical summary and its expectancy bootstrap.
- Annual/monthly `stop_rate` is repaired from replayed `initial_stop` and
  `trailing_stop` reasons: the legacy generic helper counted only literal `stop`.
  Trade results, account results and all other historical table fields are unchanged.
- Seven portable EBM models are exact copies of the frozen historical exports.
  All 17,534 predictions reconstruct exactly under sequential floating-point
  addition. Summing the same contributions in another order may differ at ~1e-17.
- New control results do not retroactively validate the earlier selection process.

`SOURCE_SNAPSHOT.json` records the origin and hash of extracted execution code.
The C++ file contains only the execution kernel, **not Lorentzian classification**.
`tests/test_execution.py` is adapted from the original `test_v3_runner.py` with
imports/config paths changed; tests add C++ parity, inference and sampling checks.
The original research source commit was `294f0f4`; local contract commit
`d50b33d` preceded computation of the new random-control outcomes.

## Numerical and platform notes

- Audited Python 3.12 environment versions: `requirements-audit.txt`.
- C++17 compile flags: `-O2 -ffp-contract=off`; no fast-math.
- Every candidate path is checked against Python and an independent fill audit;
  252 control cache paths are checked similarly. Remaining paths use the same
  compiled kernel, not independently tick-verified fills.
- Marked DD follows a conservative synthetic OHLC path; the order of highs and
  lows inside a real minute is not observed. No tick-level execution claim.
- Dependency versions can emit pandas/NumPy timedelta deprecation warnings.
  These do not alter the passing results; do not suppress unrelated warnings or
  silently upgrade dependencies and claim identical replay without verification.
- Linux/macOS compiler invocation is provided. Native Windows compiler setup,
  MT5 integration and live broker parity are not implemented.

## Repository publication

Only code, portable models and allowlisted derived evidence are uploaded.
`data/`, `runs/`, virtual environments and compiled libraries are ignored.
`results/checksums.json` verifies the published evidence; `verify_evidence.py`
also checks local Markdown links. The public visibility is the repository owner's
choice, not an assertion that the model is safe for live money.
