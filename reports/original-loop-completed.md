# Implementation loop

Status: complete on 17 October 2026. All original milestones and authorized continuation deliverables verified. First session stopped at six attempts; explicitly authorized continuation completed in four attempts, within its 90-minute/six-attempt bounds. No limits extended and no jobs running.

## Goal
Build and evaluate a reproducible local experiment predicting next-day Jack's Gelato menu snapshot flavour presence by shop. Deliver a validated parser, explicit missing-data audit, two temporal baselines, a first logistic-regression model and a reproducible evaluation report. A model that fails to improve on the baselines is a valid research result; fabricated improvement is not a success condition.

The user accepted next-day snapshot flavour presence as the target. Plan details and evidence are in implementation-plan.md; durable constraints are in global.md.

## Plan
1. Confirm target, freeze source revision and forecast-cutoff convention, and create a project-local Python environment with locked dependencies.
2. Acquire the pinned upstream data into an ignored source directory. Audit files and parse separate snapshot and appearance tables; preserve raw names and unknown/missing states.
3. Freeze chronological train/validation/test boundaries and label policy before inspecting test outcomes. Both shops stay together by target date. Enforce feature availability at issue time.
4. Implement and compare historical-frequency and persistence baselines on identical eligible forecasts.
5. Fit L2-regularized per-flavour logistic regression and compare one boosted-tree challenger, tuning only through expanding-window validation. Use documented constant predictions for single-class training labels.
6. Freeze model settings, evaluate once on the held-out period and write a report with reproducibility metadata, coverage, uncertainty and an honest model-or-baseline recommendation.

Use implementation-plan.md for schemas, split proposals, features, metrics, model-retention gate and verification cases. One conservative boosted-tree challenger is included in milestone 4 following the user's accepted model-selection approach. External features, UI, deployment and live automation remain outside this loop.

At session activation, record a wall-clock start, deadline and iteration count in log.md and memory.md. Reconcile actual files and running jobs first. Run the smallest useful milestone attempt, verify it, append the result and refresh current state before deciding whether to continue.

## Iteration conditions
An iteration is one bounded implementation-and-verification attempt at a milestone below. Failed attempts and repairs count toward the same total; retrying does not reset the budget. Measurable progress means a new durable deliverable passes its stated gate, or a failing required check is repaired and verified. Edits, repeated commands or speculative explanations alone do not count.

| Milestone | Evidence required before advancing |
|---|---|
| 0. Experiment setup | Target confirmed; source SHA and cutoff recorded; dependency lock and successful package import |
| 1. Parser and audit | Every input has a parse status; gap/duplicate/date-mismatch report; unique table keys; sampled format checks; meaningful parser and missingness tests pass |
| 2. Evaluation contract | Exact shared date splits and label policy saved; untouched test reserved; tests show post-cutoff data cannot affect features |
| 3. Baselines | Deterministic frequency and persistence predictions; same scored rows and labels; metrics and unavailable-forecast counts saved |
| 4. Logistic regression | Validation-only settings and support selection; single-class fallback verified; comparison against both baselines; configuration frozen |
| 5. Final assessment | One frozen test evaluation; coverage and novelty reporting; paired date-block uncertainty; run manifest and final recommendation |

After every attempt assess its gate, progress, elapsed time, total iterations and consecutive no-progress count. Advance only when the current gate passes. If a gate fails, isolate the cause and choose a narrower repair within the remaining limits. If a change affects the target, held-out protocol or authorized scope, record the issue and defer dependent work until resolved. Never silently switch targets or tune on the held-out result.

## End condition
Success requires all six milestone gates with recorded evidence, reproducible commands and artifacts, and a final report. Report model improvement only when the predeclared retention gate is met; otherwise recommend the validated baseline and describe the limitations.

Stop at the FIRST of:
- Success with all required evidence.
- 120 minutes elapsed from activation, including dependency and tool waits.
- Six total milestone attempts, including unsuccessful attempts and repairs.
- Two consecutive iterations without verified measurable progress.
- A user stop instruction.
- A blocking dependency or missing decision that prevents further useful authorized progress after independent work is complete.

Check remaining time before starting long work; constrain command timeouts to the remaining budget and stop managed jobs at the deadline. Never reset or extend any limit automatically. Six attempts may be insufficient if repairs are needed; that means partial progress, not permission to continue.

At every incomplete stop, preserve artifacts and resumable state, append verification and the stopping reason to log.md, refresh memory.md, and report completed milestones, evidence, remaining work and the next useful action. A stopping limit is not completion. A later explicitly authorized continuation must record its new bounds before execution; preparing this loop grants no such continuation.

## Recorded progress
1. Experiment setup: passed, dependency lock/import/CLI verified.
2. Parser attempt: failed because a missing-footer test fixture retained a valid footer.
3. Parser repair: passed, 10 tests and full data audit completed.
4. Features/splits: passed, 14 tests and frozen temporal protocol.
5. Baselines: passed, 17 tests and comparable validation scores.
6. Model validation: passed, 21 tests and frozen selection of L2 logit C=0.01.

Historical first-session stopping point: final refit/evaluation was pending. It was subsequently completed under the explicit continuation below. No automatic retry or limit reset occurred.

## Authorized continuation — 17 October 2026
Goal: finish the frozen final evaluation, uncertainty report, reusable model and a next-day forecast command.
Plan: (1) implement and synthetically test final evaluation without outcome inspection; (2) refit pre-test and evaluate the selected model once, preserving settings and baselines; (3) implement and verify saved-model next-day forecasts; (4) reconcile reports and audit every completion criterion.
Iteration conditions: each implementation/verification attempt counts, including failures and repairs. Advance on verified deliverables; retain partial artifacts on interruption. No retuning after final outcomes become visible.
End condition: verified final report, paired date-block intervals, persisted model, working forecast command and provenance. Stop earlier at 90 minutes, six continuation attempts, two consecutive no-progress attempts or user stop. Do not automatically renew these bounds. Original 120-minute/six-attempt limits above describe the completed first session; these explicit continuation limits govern this authorized session.

## Continuation completion evidence
1. Final evaluator and synthetic tests: passed, 26 tests.
2. Single frozen held-out evaluation: passed, 178 forecasts, saved model/metrics/uncertainty/report.
3. Forecasting: passed, 28 tests; 922 next-day probabilities and top-10 lists for both shops.
4. Completion audit: saved model reproduces every held-out forecast, checksums verified, cached evaluation confirmed, docs reconciled. See reports/completion-audit.md.

No further implementation iteration is required for this experiment. Keep final outcomes frozen; optional new work requires a new authorized objective/protocol.
