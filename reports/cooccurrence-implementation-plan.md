# Co-occurrence prediction experiment

## Goal
Test whether other flavours on the issue-day menu improve next-calendar-day presence predictions over the existing own-history logistic model, using identical cleaned labels, dates and support rules. This is exploratory: the historical evaluation periods and v3 name-cleaning decisions have already been inspected. No independent accuracy claim or automatic production replacement.

## Plan
Freeze this specification before computing scores. Use v3 cleaned data and original chronological windows: train through 20 April, evaluate 21 April–20 May; train through 20 May, evaluate 21 May–19 June; train through 19 June, evaluate 20 June–19 July; then an exploratory reused holdout, train through 19 July and evaluate 20 July–17 October 2026. Both shops stay on the same side of each date split.

Baseline: refit original own-history/calendar/menu-size L2 logistic model, C=0.01, minimum five positive eligible training targets, top ten, threshold 0.2. Do not compare to the all-history latest model on dates it trained on.

One fixed challenger: baseline plus up to five positively associated peer flavours per shop and target flavour. Select peers using same-day co-occurrence only on valid training menus through that fold's cutoff. Require each flavour on at least 20 training menus in that shop and pair together at least 10 times; require lift >1. Rank by descending lift, then joint count, then name. Exclude the target itself. No tuning of these rules after scores.

Add ten binary features (five peer slots per shop, zero outside that shop), recording those peers' presence on the previous calendar day. Empty slots are zero. Missing issue-day menus are excluded under baseline eligibility. Training scalers and regression coefficients fit only training targets. Later evaluation observations may enter later forecasts as lagged history, as in the existing rolling one-day-ahead protocol; peer identities and coefficients remain frozen during each window.

Report micro average precision (primary), Brier score, recall@10 over all actual flavours, vocabulary coverage, and per-shop/per-window results. Pool the three disjoint validation windows with their fold-specific vocabularies. Use 200 paired seven-calendar-day moving-block bootstrap replicates, keeping both shops together, for AP gain, Brier reduction and recall gain on pooled validation and reused holdout. These intervals do not account for prior data inspection, alias selection or all model-selection uncertainty.

Predeclared promising-result screen: pooled validation AP gain at least 0.005 with Brier no worse, AP gain in at least two of three validation windows; reused-holdout AP gain at least 0.005 with Brier no worse, and no shop's holdout AP loss exceeding 0.01. Passing merits future prospective confirmation, not deployment. Failure means retain baseline for this tested encoding; it does not prove all co-occurrence approaches fail.

## Iteration conditions
1. Implement train-only peer selection and strictly lagged features. Pass hand-counted and future-mutation leakage tests before fitting.
2. Run the one frozen paired experiment across all four windows. Save predictions, selected peers, metrics, input/code hashes and parameters; check progress after each window.
3. Calculate paired uncertainty, inspect consistency, verify old artifacts remain intact, and write findings with a keep/test-further recommendation.

## End condition
Maximum 30 minutes from first experiment execution; one fixed specification, at most two execution attempts solely for implementation faults. Stop after two consecutive verification failures or elapsed limit and report partial work. Do not reset limits or tune after results. Success requires completed comparisons, leakage tests, saved evidence and preservation checks. No new ongoing goal or background automation.
