# Does novelty separation improve prediction?

Exploratory fixed comparison, 17 October 2026. These historical periods were already inspected; this is not a new independent test. Production model and original results are unchanged.

## Known-flavour comparison

Same forecast rows, training vocabulary, C=0.01, threshold=0.2 and k=10. A standalone novelty forecast leaves existing known-flavour probabilities unchanged by definition. Status-only adds past appearance count, confirmed return count, first-observed age, present-with-no-confirmed-return indicator and returning indicator. Integrated adds an out-of-time novelty probability.

| Period | Model | Micro AP ↑ | Brier ↓ | All-flavour recall@10 ↑ |
|---|---|---:|---:|---:|
| validation | current | 0.57584 | 0.011906 | 44.136% |
| validation | status_only | 0.55493 | 0.012336 | 42.141% |
| validation | integrated | 0.55507 | 0.012339 | 42.186% |
| exploratory_final | current | 0.63862 | 0.009979 | 45.046% |
| exploratory_final | status_only | 0.62210 | 0.010312 | 43.435% |
| exploratory_final | integrated | 0.62103 | 0.010330 | 43.476% |

## New-flavour event prediction

Target: at least one flavour at this shop tomorrow never previously recorded at either shop. Simultaneous first appearances count at both shops. This predicts an event, not names. First 90 days are history-building, excluded from fitting novelty targets. “New” means first recorded, not necessarily a genuinely new recipe. Unknown snapshot days can hide introductions; conservative naming can create false novelty.

| Period | Predictor | AP ↑ | Brier ↓ | Event rate |
|---|---|---:|---:|---:|
| validation | frequency | 0.8296 | 0.1815 | 69.9% |
| validation | recent30 | 0.8199 | 0.1896 | 69.9% |
| validation | novelty_logit | 0.8514 | 0.1775 | 69.9% |
| exploratory_final | frequency | 0.7245 | 0.2043 | 59.0% |
| exploratory_final | recent30 | 0.7645 | 0.2083 | 59.0% |
| exploratory_final | novelty_logit | 0.7578 | 0.2020 | 59.0% |

## Paired uncertainty on reused final period

200 moving-block bootstrap replicates, seven calendar days, both shops sampled together, seed 42. Positive differences favour the added model. Intervals do not remove the exploratory nature of this comparison.

| Comparison | AP gain 95% interval | Brier reduction 95% interval |
|---|---|---|
| status_only_vs_current | [-0.02492, -0.00730] | [-0.000477, -0.000207] |
| integrated_vs_current | [-0.02601, -0.00794] | [-0.000509, -0.000210] |
| novelty_vs_frequency | [-0.01299, 0.06900] | [-0.005551, 0.010034] |
| novelty_vs_recent30 | [-0.03462, 0.01984] | [-0.001563, 0.018375] |

## Temporal safeguards and scope

Outer windows match the original three expanding validation windows and reused 90-day final period. Novelty models use only outer-training targets and past features. Training meta-features are generated in expanding 30-day blocks fitted strictly before each block; before enough novelty-training examples exist, a past-only smoothed frequency fallback is used. Each evaluation window uses a novelty model fitted at its outer training cutoff, with fixed weights. Neither actual target novelty nor future first-appearance dates enter features. Missing snapshots do not create artificial returns.

All model settings were fixed before this run; no hyperparameter search or production replacement. No novel names were removed from the denominator of all-flavour menu recall. Known-flavour label support stays identical to the original experiment. Results and per-shop/fold metrics are in artifacts/novelty-experiment/results.json; paired predictions and unchanged-original hashes are also saved there.

Reproduce with `.venv/bin/python scripts/novelty_experiment.py` in a fresh experiment output directory and an explicitly authorized time window. Existing output is protected against overwriting. Independent confirmation would require newly collected future data and a prospective protocol.

## Conclusion

Keep the current known-flavour model. In this prespecified implementation, both additions reduced AP in all three rolling validation folds and the reused final period. Validation AP fell from 0.57584 to 0.55507; reused-final AP fell from 0.63862 to 0.62103. Reused-final Brier worsened from 0.009979 to 0.010330, and all-flavour recall@10 fell from 45.05% to 43.48%. The exploratory paired intervals favour the current model. Most degradation came from the status/history features; adding predicted novelty on top had little incremental effect.

The standalone event forecaster is a separate output, not an automatic improvement to flavour predictions. On the reused final period, its AP was 0.7578 versus 0.7245 for shop historical frequency and 0.7645 for trailing 30-day frequency. Its Brier was 0.2020 versus 0.2043 and 0.2083. All paired novelty improvement intervals include zero. It is not established as better than these simple baselines.

Results differ by shop: new-to-recorded-history event rates were 38.6% at All Saints and 78.9% at Bene't Street on this period. The novelty classifier improved AP at All Saints (0.544 vs constant shop frequency 0.386), but not Bene't Street (0.782 vs 0.789). These subgroup results are exploratory and not separately uncertainty-tested. High pooled AP partly reflects differing shop event rates; it must not be read as 75.8% accuracy.

This does not rule out every possible new/returning model. It tests a fixed L2-logit encoding and one daily novelty head, not separately tuned cohort models or predicting novel names. Initial first-run status can remain uncertain across missing dates. No production model was replaced. Independent conclusions need new prospective data; no further tuning on the previously inspected final period was done in this run.

## Verification
32 tests pass, including future-mutation and chronological cross-fit checks. All original validation/final artifact hashes and scientific-code hashes were rechecked after execution and remain unchanged. Settings and script hashes are saved with the new experiment. Same-row truth, label order and dates were asserted against saved original predictions for every outer fold. Completed in three bounded attempts: implementation/tests; fixed experiment; integrity audit and interpretation.
