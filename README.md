# Jack's flavours

A reproducible local experiment forecasting flavour presence in the next Jack's Gelato menu snapshot, by shop.

## Run the shared model

This repository includes the trained model, audited processed history, evaluation artifacts and charts. After cloning/downloading it, open a terminal in the project directory and run:

```sh
uv sync --locked
uv run python -m jack_flavours forecast --model latest
```

The lockfile requires Python 3.13; `uv` can provision it. The included history ends on **17 October 2026**, so this reproduces the **18 October 2026** forecast. It does not fetch current menus automatically. Results are written to `artifacts/forecasts/latest/`.

The trained weights are in `artifacts/latest/model.joblib`. Keep the source, configuration, dependency lock and `data/processed/` alongside them: inference reconstructs lagged features from that history. Load joblib bundles only from a trusted source, as loading them can execute Python code.

See [data provenance and attribution](DATA_PROVENANCE.md). To explore the charts, open `artifacts/flavour-statistics/index.html` locally after downloading the repository.

## Final results
The initial implementation and final evaluation are complete. The held-out period was **20 July–17 October 2026**, with 178 eligible shop/day forecasts. All model settings were selected beforehand on three expanding-window validation periods.

| Predictor | Held-out micro average precision ↑ | Brier error ↓ | Top-10 all-flavour recall ↑ |
|---|---:|---:|---:|
| Historical frequency | 0.2316 | 0.01642 | 18.1% |
| Previous-day persistence | 0.5322 | 0.01025 | 44.5% |
| L2 logistic regression | **0.6386** | **0.00998** | **45.0%** |

Retained model: **L2 logit, C=0.01**. The ranking improvement over persistence is supported by the paired seven-day block-bootstrap interval. The small Brier improvement is uncertain; top-10 recall is almost unchanged. AP is not classification accuracy. Trained labels cover 63.7% of held-out flavour appearances; new/unsupported flavours remain a substantial limitation.

Read the [final report](reports/final-evaluation.md), [validation report](reports/model-validation.md) and [completion audit](reports/completion-audit.md).

## Technical guide: what the prediction model does

### The outcome and unit of observation
For each flavour, one observation is a **shop on a target date**. The outcome is binary: `y = 1` if that flavour appears in the valid menu snapshot, otherwise `y = 0`. A separate model is fitted for each supported flavour, pooling observations from both shops and including a shop indicator. Multiple flavours can be present together, so this is **multi-label binary classification**, not multinomial logit: the flavour probabilities need not sum to one.

Only flavours with at least five positive eligible training observations are modelled. Training and evaluation rows require valid snapshots on both the target date and the previous calendar day. Missing or quarantined menus are unknown, not evidence of absence. This predicts recorded menu presence, not sales, demand, quantities or all-day availability.

### Current predictor: L2-regularised binary logit
The forecast uses scikit-learn's `LogisticRegression`, with `StandardScaler` fitted on training data, the `lbfgs` solver, `max_iter=1000`, and **C = 0.01**. There is no class weighting. Both the preserved July model and the latest October refit use this specification.

For a flavour, the model has the form:

```text
p = P(y = 1 | x) = 1 / (1 + exp(-z))
z = intercept + β₁x₁ + … + β₂₁x₂₁
log(p / (1 - p)) = z
```

“Logit” is the log of the odds, `log(p / (1-p))`. The model is linear in log-odds, while the sigmoid converts that score to a probability between zero and one. For example, a probability of 0.8 corresponds to odds of 4 to 1.

**Regularisation** penalises large coefficients when fitting the model. L2 uses squared coefficient magnitudes, conceptually minimising logistic prediction loss plus a penalty proportional to `Σ βⱼ²`. This shrinks estimates towards zero and can reduce overfitting when flavours have few appearances or predictors overlap. It generally does not set coefficients exactly to zero. In this implementation, smaller `C` means stronger regularisation; the precise penalty scaling follows the library implementation. Validation compared `C ∈ {0.01, 0.1, 1, 10}` and selected 0.01.

For a continuous feature, a one-training-standard-deviation increase changes log-odds by its fitted coefficient, holding other inputs fixed; `exp(β)` is the corresponding odds ratio. This is **not** a constant percentage-point change in probability: the probability effect depends on the starting probability. The scaler also transforms binary indicators, so a raw 0-to-1 change must account for their scale. These are penalised predictive associations, not causal effects or conventional unpenalised estimates with reported standard errors. The project does not estimate p-values or causal effects of season or shop.

### What goes into the model?
There are 21 features per flavour/shop/date:

| Feature group | Inputs | What it helps capture |
|---|---|---|
| Calendar | Sine/cosine of weekday and month; weekend flag | Repeating weekly and seasonal patterns; sine/cosine keeps adjacent cycle endpoints close |
| Shop and menu | Shop indicator; previous day's menu size | Differences between locations and recent menu capacity |
| Recent presence | Presence 1, 2 and 7 calendar days earlier; missing flags for lags 2 and 7 | Persistence and weekly recurrence |
| Recent frequency | Observed appearance rates over 7, 30 and 90 calendar days, plus valid-observation counts in each window | Shorter and longer patterns, with information about data coverage |
| Recency and runs | Days since last observed presence; never-seen flag; consecutive observed days present | Time off the menu and ongoing runs |

All menu-derived inputs for target day `t` use observations through `t−1` only. Calendar features describe the already-known target date. Rolling rates exclude missing observations from their denominators; a gap breaks a streak. Unavailable lags/rates use zero placeholders with missingness or coverage information; never-seen recency uses 999 with a separate flag. This is an autoregressive-style classifier with calendar covariates, rather than a survival model directly estimating time until removal or return.

### Tree challenger: histogram gradient boosting
We also tested scikit-learn's **`HistGradientBoostingClassifier`**, separately for each flavour. It builds an additive sequence of small decision trees, with each boosting step improving the current classification loss. Histogram binning groups feature values to make split searches efficient. Trees can learn nonlinear thresholds and interactions, such as recency behaving differently by shop, without specifying those interactions in the logit equation.

The fixed challenger settings were:

| Setting | Value | Role |
|---|---:|---|
| `max_iter` | 100 | Number of boosting iterations |
| `max_leaf_nodes` | 7 | Limits each tree's complexity |
| `learning_rate` | 0.05 | Shrinks each tree's contribution |
| `min_samples_leaf` | 20 | Discourages splits supported by very few rows |
| `l2_regularization` | 10.0 | Penalises leaf values |
| `early_stopping` | False | Uses the fixed iteration budget without an internal stopping split |
| `random_state` | 42 | Reproducibility |

The tree model is **not used by the current forecast**. On pooled chronological validation, logit achieved micro average precision **0.5758**, versus **0.5615** for boosting; both Brier scores round to 0.0119. Boosting beat logit's AP in only one of three windows and did not meet the predeclared replacement rule. That rule required at least a 0.01 pooled AP gain, no worse Brier error, and gains in at least two windows and both shops when logit already passed the baseline gate. This supports choosing logit for this experiment, not a general claim that logit always beats trees. See [validation evidence](reports/model-validation.md).

### How we avoid looking ahead and interpret the scores
We used three expanding-history validation windows: **21 April–20 May**, **21 May–19 June**, and **20 June–19 July 2026**. Each model trained only through the day before its window. Vocabulary, scaling and coefficients were fitted using training rows only; both shops share the same date boundaries. Within each window, weights stay fixed while lagged features update as earlier menus become available. These are successive one-day-ahead predictions, not a forecast of an entire month made on its first day. Historical collection availability is assumed rather than timestamp-verified.

Model settings were chosen on those windows, then frozen before the **20 July–17 October** held-out assessment. The baselines were training-history frequency by shop/flavour and previous-day presence. Paired seven-day date-block bootstrapping compares models while retaining some short-run dependence; its uncertainty concerns forecast performance, not individual logit coefficients.

- **Micro average precision (AP):** measures how well probabilities rank actual appearances above absences, pooling supported flavour/forecast pairs. Higher is better; 0.6386 is not “63.86% accuracy”.
- **Brier error:** the mean of `(p − y)²`; lower is better. It evaluates probability error, but many absent flavours can make the average look small.
- **Top-10 recall:** the fraction of the actual menu captured by the ten highest-ranked flavours, averaged across forecasts. Unsupported flavours count as misses. It is not the fraction of our ten suggestions that are correct—that is precision@10.
- **Calibration:** asks whether, for example, outcomes assigned roughly 70% probability occur roughly 70% of the time. Strong ranking does not guarantee calibrated probabilities.

The selected probability threshold is **0.2**, while the displayed shortlist takes the **top ten**. These are separate decision rules, chosen using validation F1 and menu F1 respectively; neither changes the fitted probabilities. A flavour can enter the top ten without exceeding the threshold, or exceed the threshold without entering the top ten.

The latest refit uses history through 17 October with the same settings. It supports 526 labels versus the July model's 461. Because it now trains on the earlier held-out period, the old test scores do not independently validate this updated model. It still cannot name an unseen flavour. The exploratory new/returning-flavour separation did not improve results and is not part of the current predictor.

### Where to read the implementation

- [models.py](src/jack_flavours/models.py): logit, tree settings integration and model-selection rule.
- [features.py](src/jack_flavours/features.py): target eligibility, all 21 features and chronological splits.
- [experiment.json](configs/experiment.json): frozen candidate settings and evaluation configuration.
- [evaluation.py](src/jack_flavours/evaluation.py): metrics and shortlist/threshold selection.
- [final.py](src/jack_flavours/final.py), [refit.py](src/jack_flavours/refit.py) and [forecast.py](src/jack_flavours/forecast.py): evaluation fit, latest-history refit and inference.

## Forecasts
The default forecast uses a latest-history refit trained through **17 October 2026**, including the last three months plus the earlier history. Forecasts for **18 October 2026** are saved locally:

- Full probabilities: `artifacts/forecasts/latest/forecast-2026-10-18.csv` (526 flavours × two shops).
- Top ten per shop: `artifacts/forecasts/latest/forecast-2026-10-18-top.csv`.
- Provenance: `artifacts/forecasts/latest/forecast-2026-10-18.json`.
- Latest model: `artifacts/latest/model.joblib`.
- Preserved evaluation model: `artifacts/final/model.joblib`, trained through 19 July. Its forecasts remain under `artifacts/forecasts/`.

[July versus latest forecast comparison](reports/latest-refit-comparison.md): All Saints retains seven of its top ten; Bene’t Street retains all ten. The latest refit uses unchanged settings and is not independently evaluated; the held-out scores above apply only to the July model.

The model predicts a snapshot, not all-day stock. Its probabilities are not perfectly calibrated. It cannot name new flavours. Forecasting is anchored to the supplied history date, not automatically to today's date, and no live data feed is configured. New source data requires an audited update; this experiment checks pinned history hashes.

## Local commands
Dependencies are locked in uv.lock. From this directory:

```sh
uv sync --locked
.venv/bin/python -m pytest -q
.venv/bin/python -m jack_flavours --help
.venv/bin/python -m jack_flavours forecast
.venv/bin/python -m jack_flavours forecast --issue-date 2026-10-17
.venv/bin/python -m jack_flavours forecast --model evaluation
.venv/bin/python scripts/refit_latest.py
.venv/bin/python -m jack_flavours evaluate-final
.venv/bin/python scripts/verify_artifacts.py
```

`evaluate-final` returns checksum-verified saved results after the first run; it does not retrain or rescore. `verify_artifacts.py` reproduces every saved held-out prediction from the serialized model and checks output integrity. `forecast` loads the latest model when available (otherwise the evaluation model) without refitting, discards post-issue observations, and excludes shops without valid issue-day snapshots.

`scripts/refit_latest.py` manually creates or reuses the latest audited-history refit and regenerates the comparison. It does not fetch new source data.

The earlier stages are `audit`, `freeze`, `baselines` and `validate-models`. The model-selection command refuses to overwrite frozen settings. Do not rerun development to tune against final outcomes. A new experiment needs a new bounded plan and a new evaluation protocol.

For a fresh data checkout, acquire https://github.com/EdmundGoodman/jacks-menu-history into data/source and check out `fba48c936c059625f399353e5afa8b071fc51457` before audit. Do not run upstream workflows. The audited processed history, saved models and selected research artifacts are included for reproducibility. The raw upstream checkout, local environments, credentials and assistant working notes are excluded. This is a research repository, not a deployed live service.

## Evidence and navigation
36 tests pass, covering parsing, missingness, training-only preprocessing, future-mutation invariance, chronological splits, model fallback/selection, paired resampling and saved-model inference. The serialized model reproduces all 178 held-out forecasts exactly within numeric tolerance.

- [Data audit](reports/data-audit.md): 1,417 source files; 1,222 usable, 195 quarantined, 71 expected shop/days missing.
- [Evaluation protocol](reports/evaluation-protocol.md), [baselines](reports/baselines.md).

Source: [Edmund Goodman’s menu-history repository](https://github.com/EdmundGoodman/jacks-menu-history).

## Novelty-separation follow-up
An exploratory test of new/returning status and predicted novelty did not improve the known-flavour model. Final-period AP fell from 0.639 to 0.621, with worse Brier and menu recall. The standalone novelty forecast did not establish an advantage over simple baselines. The original model is unchanged. [Full comparison](reports/novelty-experiment.md). The current test suite has 36 passing tests.

## Flavour statistics gallery
Explore [the local chart gallery](artifacts/flavour-statistics/index.html): most common flavours, mean/median menu stays and return gaps, per-shop charts for common and all flavours, and downloadable event-level data. Regenerate with `.venv/bin/python scripts/flavour_descriptives.py`. The gallery also includes within-shop menu similarity at calendar-day lags 0–180 (mean Jaccard overlap), with valid-pair counts and CSV downloads. Descriptive matching removes standalone `and` and `&`, preserving raw titles and writing `data/cleaned-descriptive-v3/` (including the three user-confirmed ingredient-order aliases); `title-mapping.csv` records every mapping. Frozen model inputs, model labels and published evaluation results retain their original normalization; these charts do not imply a model refit. Missing menus and unfinished spells are explicitly distinguished from observed absences.


## Data-quality audit

[Cleanlab audit report](artifacts/cleanlab-audit/index.html) distinguishes automated flags from supplementary naming candidates and inspected evidence. Reproduce with `scripts/cleanlab_audit.py --root PROJECT_PATH --out OUTPUT_PATH` using an environment with Cleanlab 2.9.0 and its Datalab extra, pandas, NumPy, scikit-learn and PyArrow. The audit is read-only and does not merge recipes or retrain models.


[Cleaning comparison](artifacts/cleaning-comparison-v3/index.html): original, conjunction-only and three-confirmed-alias statistics and fixed latest-history refits. Default forecast still uses the preserved original latest model; new comparison forecasts are under `artifacts/cleaning-comparison-v3/confirmed/`. No independent accuracy improvement is claimed. `scripts/cleaning_comparison.py` reproduces numerical results in a fresh output directory and refuses to overwrite existing refits.

The summary gallery includes within-shop flavour co-occurrence: frequent pairs, support-filtered lift, both conditional probabilities and a downloadable pair-level CSV. Regenerate through `scripts/flavour_descriptives.py`.


[Co-occurrence prediction experiment](reports/cooccurrence-experiment.md): the fixed lagged-peer augmentation did not improve the cleaned baseline; original models retained. [Predeclared implementation plan](reports/cooccurrence-implementation-plan.md).

## Daily prospective forecasts

The daily pipeline evaluates forecasts issued before the target day, starting with the saved 18 October 2026 forecast. It collects dated menu snapshots, scores immutable predictions, refits the fixed-settings logistic model using v3 cleaned names, and saves the following day's forecasts alongside a frozen model and persistence comparator.

```sh
uv sync --locked
uv sync --locked --project integrations/menu-collector
uv run --locked python -m jack_flavours.daily run --state daily-state
```

The command uses today's Europe/London date and writes separate live state; it never changes the frozen experiment. Maximum three collection attempts per day, with stale or missing menus excluded. Manual runs count toward that limit. A failed refit preserves scores but does not issue a new adaptive forecast. The workflow is capped at 30 minutes and serializes data writes.

The public `daily-data` branch contains the permanent observations, forecasts, accuracy records and static dashboard under `site/`. V3 models are versioned release assets named `v3-YYYY-MM-DD.joblib`. Original-name forecasts and the 18 October score remain separately labelled in the archive. V3 prospective accuracy starts with 19 October outcomes. Model bundles are Python pickle-based artifacts: load only trusted, checksum-verified versions from this project. The dashboard reports top-ten hit rate, all-menu coverage, missing shops and shared-label probability comparisons.

Hosting uses **free GitHub Actions on a public repository and Cloudflare Pages Free**. No Workers paid plan, paid runner, database or LLM API is required. The workflow refuses to run on a private repository. Cloudflare Pages should deploy only `daily-data`, with no build command and output directory `site`; disable preview builds. Use its free `pages.dev` address. Instructions: [deployment guide](reports/daily-deployment.md).

See [implementation plan](reports/daily-learning-implementation-plan.md) for the evaluation policy, provenance and bounded runtime. Original validation/test results do not validate the daily refits.
