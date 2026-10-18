# Model validation results

These results use rolling validation only. The final 90-day test has not been evaluated; they are model-selection evidence, not an unbiased final performance estimate.

Selected for the later final evaluation: **logit**. Best logistic C: **0.01** (smaller C means stronger L2 regularisation).

| Predictor | Pooled micro AP ↑ | Brier ↓ | k | Threshold |
|---|---:|---:|---:|---:|
| frequency | 0.2160 | 0.0178 | 10 | 0.2 |
| persistence | 0.4505 | 0.0135 | 10 | 0.8 |
| logit | 0.5758 | 0.0119 | 10 | 0.2 |
| boosting | 0.5615 | 0.0119 | 8 | 0.35 |

## Stability across validation periods

| Period | Predictor | Micro AP | Brier | Menu recall@k including unseen flavours |
|---|---|---:|---:|---:|
| validation_1 | frequency | 0.1718 | 0.0197 | 0.1913 |
| validation_2 | frequency | 0.3165 | 0.0160 | 0.2286 |
| validation_3 | frequency | 0.2049 | 0.0177 | 0.1917 |
| validation_1 | persistence | 0.5034 | 0.0124 | 0.4802 |
| validation_2 | persistence | 0.4732 | 0.0127 | 0.4498 |
| validation_3 | persistence | 0.3760 | 0.0154 | 0.4082 |
| validation_1 | logit | 0.5983 | 0.0118 | 0.4507 |
| validation_2 | logit | 0.6061 | 0.0114 | 0.4560 |
| validation_3 | logit | 0.5208 | 0.0126 | 0.4156 |
| validation_1 | boosting | 0.6011 | 0.0116 | 0.4165 |
| validation_2 | boosting | 0.5929 | 0.0112 | 0.4027 |
| validation_3 | boosting | 0.4855 | 0.0128 | 0.3573 |

## Decision evidence

```json
{
  "selected": "logit",
  "baseline_fallback": "persistence",
  "logit_passes_validation_gate": true,
  "boosting_passes_validation_gate": true,
  "boosting_substantially_beats_logit": false,
  "boosting_better_folds": 1,
  "boosting_better_both_shops": false,
  "final_test_status": "not_evaluated"
}
```

## Regularisation comparison

| C | Validation micro AP | Brier |
|---|---:|---:|
| 0.01 | 0.5758 | 0.0119 |
| 0.1 | 0.5480 | 0.0120 |
| 1 | 0.4511 | 0.0135 |
| 10 | 0.3325 | 0.0171 |

## Coverage and limitations

| Period | Eligible forecasts | Supported labels | Coverage of actual flavour appearances |
|---|---:|---:|---:|
| validation_1 | 60 | 410 | 68.8% |
| validation_2 | 60 | 434 | 69.0% |
| validation_3 | 56 | 449 | 70.3% |

Flavour vocabulary requires at least five positives in eligible training snapshots. Unknown and low-support flavours are not forecast, but count as misses in all-flavour menu recall. Canonical names are intentionally conservative; spelling variants may fragment labels. Brier scores average many negative labels, so read them alongside AP and calibration. Historical collection availability is assumed, not verified. Quarantined dates and missing inputs reduce coverage.

Model selection and operating points used these validation windows, so the reported advantage may be optimistic. Per-shop metrics, calibration bins and per-flavour results (including training support) are saved in ignored artifacts/validation. No semantic alias tuning or test-driven parameter changes were made.

## Remaining work

The six-attempt session limit was reached after validation (one parser-test fixture repair consumed an attempt). Final refitting, held-out scoring, paired date-block uncertainty and a deployable model artifact remain pending a separately authorized continuation. The frozen selection and predictions are saved; do not retune them on the final test.

## Interpretation

L2 logit wins the prespecified ranking/probability selection rule, not every practical metric. Weighted all-flavour menu recall at the validation-selected k is frequency: 20.4%, persistence: 44.7%, logit: 44.1%, boosting: 39.3%. Logit and persistence both select k=10, so their menu recall is directly comparable. The current evidence supports better ranking/probability estimation, but not an improvement in top-10 menu recall. Boosting selected k=8, so its recall uses fewer predictions. Final-test evidence and uncertainty remain required.

## Subsequent completion

The pending final assessment described above was completed in an explicitly authorized continuation on 17 October 2026. See [the final evaluation](final-evaluation.md). Validation metrics and selection remain unchanged.
