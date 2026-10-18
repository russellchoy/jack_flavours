# Does co-occurrence improve prediction?

**Recommendation: retain the baseline. This fixed co-occurrence encoding made predictions slightly worse.** The predeclared promising-result screen failed. This does not rule out every possible use of menu context.

## Implementation
Compared the same cleaned own-history logistic baseline against that baseline plus up to five positively associated peer flavours per shop. Peer associations were learned only from each window’s training menus. Each flavour needed at least 20 appearances, each pair at least ten shared menus, and lift above one. The challenger used selected peers’ presence on the previous calendar day, never the target-day menu. Regression settings stayed fixed at C=0.01, minimum target support five, top ten and threshold 0.2.

The full plan was written before scores were computed. One specification was run once across three chronological validation windows and the reused July–October holdout, without result-driven tuning. Both models were refitted at each training cutoff; the all-history model was not evaluated on its training dates.

## Results

| Evaluation | Metric | Baseline | With peers | Change |
|---|---|---:|---:|---:|
| Three pooled validation windows | Average precision ↑ | 0.57487 | 0.57205 | −0.00281 |
| Three pooled validation windows | Brier score ↓ | 0.012059 | 0.012162 | +0.000104 |
| Three pooled validation windows | Recall@10 ↑ | 44.39% | 43.93% | −0.46 pp |
| 20 July–17 October, reused | Average precision ↑ | 0.65951 | 0.65522 | −0.00429 |
| 20 July–17 October, reused | Brier score ↓ | 0.010093 | 0.010265 | +0.000172 |
| 20 July–17 October, reused | Recall@10 ↑ | 47.56% | 46.97% | −0.59 pp |

Recall@10 uses all actual flavours in each menu, including unsupported labels. Average precision and Brier are calculated over the identical supported labels for each paired comparison.

Average precision improved in only one of the three validation windows:

| Validation dates | Baseline AP | Peer AP |
|---|---:|---:|
| 21 April–20 May | 0.59923 | 0.59138 |
| 21 May–19 June | 0.60207 | 0.60523 |
| 20 June–19 July | 0.52137 | 0.51730 |

In the reused holdout, both shops worsened: All Saints AP 0.69541 → 0.68815; Bene’t Street 0.62955 → 0.62792. Both shops also had higher Brier scores and lower recall@10.

## Uncertainty and scope
Used 200 paired seven-calendar-day moving-block bootstrap replicates, keeping shop observations together. Reused-holdout AP change had a descriptive 95% interval of **−0.00687 to −0.00073**; recall@10 change **−0.93 to −0.28 percentage points**. Validation AP change interval crosses zero (−0.00710 to +0.00157). These intervals do not remove the effects of earlier data inspection, frozen full-history-informed name-cleaning decisions, or model-selection uncertainty.

Only **92 of 460 supported flavours** had any eligible peers in the reused-holdout training data; eligibility rose from 76/408 to 89/448 over the three validation windows. Thus this tests a conservative same-day-lift peer encoding, not every flavour’s full menu context. Shared seasonality and menu runs may explain strong descriptive co-occurrence without adding stable information beyond the existing history/calendar features; this is a possible explanation, not something established by this experiment.

The previously reported uncleaned-model AP of 0.6386 is not the comparator here: both arms use the same v3 cleaned targets and support rules. The historical period has already been used and inspected, so the results are exploratory. No new independent accuracy claim is made.

## Verification and artifacts
44 tests pass, including future-mutation checks for peer selection and target-day feature exclusion, shop separation and missing issue-day handling. Original processed data, frozen validation/final artifacts and latest model hashes were verified unchanged. No production forecast or model was replaced.

Numerical results, paired prediction matrices, selected peers, plan/code/data hashes and parameters are stored in `artifacts/cooccurrence-experiment/` in the project. Implementation: `src/jack_flavours/peer_features.py`; execution: `scripts/cooccurrence_experiment.py`. The script refuses to overwrite an existing experiment directory. The execution took about 16 seconds within its preset 30-minute limit.
