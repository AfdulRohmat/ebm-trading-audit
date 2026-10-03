# Publication audit report — 2026-10-03

## Delivered

- Separate EBM repository; existing research/Lorentzian repositories unchanged.
- Indonesian README explaining the model, indicators, target, chronological
  fitting, execution, costs, controls, uncertainty and production limitations.
- Seven figures generated from committed evidence, not hand-drawn outcomes.
- Seven frozen portable models, exact feature/runner source snapshots, training
  recipe, replay scripts, 240 account rows and complete conditional-control tables.
- 2,000 random-entry and 2,000 random-direction draws; 2,000 five-session block
  resamples for each cost arm. No model/feature/threshold tuning.

## Verification and engineering loop

1. Contract `d50b33d` committed before new conditional-null outcomes were inspected.
2. All 17,534 frozen predictions reproduced exactly. All primary trade fills,
   exit timestamps/reasons, net R and account returns/DD matched previous v3.
3. Candidate paths compared with both the Python implementation and independent
   fill audit; 252 distributed random-cache paths similarly audited.
4. 44 tests pass: stops, gaps, next-bar trailing, early close, no future features,
   compounding/insolvency, compiled parity, feature bins and random matching.
5. Second complete replay compared 29 artifacts: 27 byte-identical. The only two
   differences are reviewed annual/monthly stop-rate reporting corrections;
   candidate trades, controls, bootstrap and account outcomes are byte-identical.
6. 22 evidence SHA256 checks and all 24 local Markdown links pass. Plots inspected;
   source-distribution wording corrected to pooled test-grid visualization.
7. All 253 tracked/untracked source files covered by the preservation snapshot
   stayed unchanged. No raw Kaggle data, full event matrix, credential or home
   directory path is included in the publication allowlist.

Tests emit existing pandas/NumPy timedelta deprecation warnings. The audit does
not hide them; the warnings are compatibility work, not a failed result assertion.

## Decision for discussion

The unchanged primary runner is historically positive (+31.68R, PF 1.188 at
base cost). The classifier-independent-exit diagnostic is above roughly 97% of
matched random-entry samples, an encouraging conditional result. It does not
establish a pristine, selection-adjusted trading edge.

Buy-and-hold outperforms the 1%-risk account in total return, with a different
exposure/risk profile. The historical bootstrap still includes losses. Model
MSE is slightly worse than a training-mean baseline, only 4/7 years are positive,
and trading activity is low and uneven. Required notional leverage is material.

**Decision: publish as an auditable historical candidate, not an EA-ready or
live-money-approved strategy.** Preserve the frozen rules for broker-feed
validation and genuinely new evidence; do not optimize from these charts.
