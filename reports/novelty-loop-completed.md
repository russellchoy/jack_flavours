# Novelty experiment loop

Status: complete after three attempts on 17 October 2026. Authorized by user instruction “Test it” at 18:22:47 UTC; completed within the 60-minute/six-attempt bounds. Original completed loop preserved in reports/original-loop-completed.md.

## Goal
Determine whether separating new-flavour arrivals from known-flavour prediction improves the existing model, and whether a separate novelty event forecast is useful. This is exploratory analysis, not a fresh untouched test.

## Plan
Use the same three chronological validation folds and previously inspected final period. Keep current model, source and frozen artifacts unchanged. Compare current probabilities against L2 logit with flavour-history status features, and those features plus a past-only predicted novelty probability. Forecast first-observed-across-both-shops events separately against historical-frequency and trailing-30-day baselines. Fix C=0.01 and all settings before new results. Use a 90-day history burn-in. Generate training novelty probabilities with expanding 30-day blocks, fitted only before each block; use an outer training-only novelty model for each evaluation window. Never feed actual target-day novelty to the flavour model.

## Iteration conditions
1. Implement isolated experiment and verify novelty labels, status features and temporal cross-fitting with synthetic future-mutation tests.
2. Run fixed comparisons, save paired forecasts, metrics, per-shop results and seven-calendar-day paired bootstrap intervals.
3. Audit preservation of existing model artifacts, interpret improvements and limitations, and hand over a reproducible report.
Each failed attempt or repair counts toward six attempts. Continue only after verified progress; assess each mini-goal before advancing.

## End condition
Success: reproducible model comparison, standalone novelty evaluation, leakage checks and an evidence-based answer. Stop at the first of 60 minutes (19:22:47 UTC), six attempts, two consecutive no-progress attempts, success or user stop. Preserve partial work and report remaining criteria at any limit; never reset or extend automatically. Do not change the production candidate based on this exploratory comparison. Fresh future data is required for independent confirmation.

## Completion evidence
1. Isolated experiment and temporal tests: 32 passed.
2. All four windows and paired intervals saved under artifacts/novelty-experiment.
3. Original artifacts/scientific code preserved, report reviewed and recommendation recorded: retain current model. No further iteration is needed for this fixed comparison.
