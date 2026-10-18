# Validation baselines

Final test remains untouched. All results are on the three rolling validation windows.

| Predictor | Micro AP (higher is better) | Brier (lower is better) | Chosen k | Threshold |
|---|---:|---:|---:|---:|
| frequency | 0.2160 | 0.0178 | 10 | 0.2 |
| persistence | 0.4505 | 0.0135 | 10 | 0.8 |

Both predictors use the same forecast rows and training-only vocabulary in each fold. Missing target or previous-day snapshots are not scored. Threshold and k are selected on validation only. Precision/recall at k counts out-of-vocabulary flavours as missed. Full fold/shop metrics and calibration bins are in local artifacts/baselines/metrics.json.

| Fold | Forecasts | Eligible labels | Actual flavour appearances | In-vocabulary appearances |
|---|---:|---:|---:|---:|
| validation_1 | 60 | 410 | 750 | 516 |
| validation_2 | 60 | 434 | 754 | 520 |
| validation_3 | 56 | 449 | 697 | 490 |

Macro AP omits labels without validation positives and reports the eligible count. Empty target recall is defined as zero (valid empty menus are excluded by the parser). Stable lexical ordering breaks probability ties. No model claims are made from these baseline results.
