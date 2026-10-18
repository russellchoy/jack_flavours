# Daily forecasting and prospective evaluation

Prepared 18 October 2026. Status: proposed implementation, not activated or deployed. This document authorizes no scheduled work. The requested scoring start is 18 October 2026.

## Outcome

Automatically collect each shop's daily menu, score the forecast issued the preceding day, append the result to permanent history, refit with newly observed data, and publish the following day's predictions. Show whether daily refitting improves on a frozen model and a persistence baseline. Retraining is adaptation, not a guarantee of improvement.

## Architecture and hosting

- GitHub Actions runs the Python pipeline on a standard Linux runner. Use a new workflow written for this project, not the upstream scraping workflow.
- A dedicated `daily-data` branch stores append-only raw observations, normalized observations, forecasts, scores, manifests and run statuses. Update it with a single serialized writer and an atomic Git commit; never force-push it.
- Versioned GitHub release assets hold model bundles, with checksums referenced from the data branch. Upload and verify assets before committing manifests that reference them. Reconcile orphaned assets on retry. Keep every issued model initially and review storage after 90 days; do not silently delete evidence.
- GitHub Pages hosts a generated static accuracy/forecast dashboard. Publish only after the data commit succeeds. The data history remains authoritative if dashboard publication fails.
- Workflow artifacts and caches are temporary conveniences, never the sole permanent record. Cloudflare is optional later; it is not required for the first version.

GitHub standard hosted runners are free for public repositories; storage has separate limits. Schedules can be delayed and inactive public repositories can have schedules disabled after 60 days. See [billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions) and [schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

## Collection and daily timing

Use [jacks-menu](https://github.com/EdmundGoodman/jacks-menu), inspected at commit `ccdfca95d4eef554052ebab63d0c9733bc96624f`, as a pinned dependency in the project lock. Verify its MIT license and retain attribution when packaging. Its normal downloader uses known Google document IDs; Selenium is optional for ID discovery. Keep the existing upstream history as a separately identified historical/recovery source.

Proposed schedule: attempts at 11:37, 13:37 and 16:37 Europe/London, with the last being the daily collection deadline. Confirm publication timing during initial live checks; these times are provisional, not a claim about when Jack's updates its menu. Implement DST explicitly: use a supported timezone-aware schedule if available, otherwise UTC triggers covering both offsets with a London-time eligibility guard. A delayed job must check the actual local date/time before writing a forecast.

For each shop, accept the first valid current-day menu collected in this window and freeze it as that day's evaluation snapshot. This predicts that snapshot, not every flavour offered throughout the day. Keep later corrections as versioned records, never silently replace the original outcome.

Download into a unique attempt path: the scraper reuses an existing file instead of downloading again. Validate the date inside the menu, shop identity, parse completion, nonempty flavour list and duplicates. Reject stale or malformed menus. Store UTC fetch time, London target date, document ID, raw text hash, scraper version and validation result. Identical consecutive menus are allowed if the source date is current. Do not infer freshness from a filename or download success.

Process each shop's valid observation independently for scoring. Refit/issue once both shops are resolved, or at the final attempt with partial coverage. A shop without a valid issue-day menu gets no new forecast. No missing observation becomes a negative label. Cap collection at three scheduled attempts per London day, with request timeouts and a 30-minute workflow limit; no unbounded polling.

## Daily transaction

1. Load and verify committed state. Determine which dates/shops need work. An identical rerun must not duplicate scores or overwrite forecasts.
2. Collect and validate observations. Preserve rejected attempts with reasons.
3. Score the immutable forecast issued on the preceding London calendar day. Check its timestamp, target date, model hash and training cutoff. Do this before refitting.
4. Append valid observations to a separate live history derived from the original audited dataset. Preserve frozen experiment inputs and artifacts byte-for-byte.
5. Fit the existing L2 logistic regression with fixed C=0.01, minimum support five, existing features and operating point. Use all eligible history through today; never train on tomorrow or label missing dates absent.
6. Generate tomorrow's probabilities/top ten from the daily-refit model, frozen 17 October model and persistence baseline, using the same issue-day observations.
7. Save models and immutable forecasts, calculate daily/rolling summaries, commit state, and publish the dashboard.

Persist completed scoring even if refitting fails. Mark that run partial and preserve the previously published forecast. A fallback forecast may use the last good model only if current-day features validate; label the fallback/model age explicitly. Never manufacture a retrospective forecast and count it as prospective evidence.

## Initial model and naming policy

Start with the current production normalization and latest model, which uses the original label scheme. The separate v3 cleaned experiment is not yet integrated into the default forecast pipeline. Changing naming and daily refitting simultaneously would complicate comparison with the saved 18 October forecast.

Freeze and version normalization at launch. Preserve raw labels. A later switch to the user-approved aliases must be an explicit pipeline version with matched training/inference normalization and a documented scoring boundary. Do not add new inferred aliases automatically.

## Metrics and records

Primary measure: precision@10, displayed as “top-ten hit rate”. Store hits and number predicted, not just a percentage. If seven of ten predictions occur, the rate is 70%.

Also store recall@10 against the entire observed menu, unsupported actual flavours and coverage, average precision, and Brier score over the model's declared supported label set. Brier scores from different label universes are not directly comparable: provide a shared-label comparison across competing models and always expose coverage. For persistence top ten, rank yesterday's flavours using prior-history frequency with a deterministic tie break; if fewer than ten exist, report the actual denominator. Persistence probabilities on the shared label set are yesterday's binary presence.

Report per-shop rates and a pooled daily hit rate (sum of hits divided by sum predicted). Include the number of scored shops, missing-data status, seven-day and 30-day pooled rates, and cumulative rates since 18 October. Missing days remain gaps; zero denominators produce unavailable metrics, not zero. Use equal-date/shop paired comparisons for model comparisons.

Minimum immutable record groups:

- Observations: shop/date, snapshot ID, source date, fetch time, raw/normalized content hashes, normalization version, validity and source.
- Forecasts: issue/target dates, creation time, shop, flavour probabilities/ranks, model ID, training cutoff, supported labels, input hashes and pipeline version.
- Scores: forecast ID, observation ID, metric version, counts/denominators, scores, unsupported count and scoring timestamp.
- Runs: attempts, stage outcomes, error reason, state commit and dashboard publication status.

Corrections append a superseding score version tied to the revised observation; retain the originally reported result and visibly identify revisions.

## Bootstrap on 18 October

Import `artifacts/forecasts/latest/forecast-2026-10-18.*` as the official initial forecast. Its metadata records creation at 2026-10-17 21:12:08 UTC and training through 17 October. Verify CSV/model checksums before importing.

Fetch and validate 18 October observations if still available; otherwise use dated archived observations with their provenance and recovery timestamp. If no reliable observation exists, mark the date unscored. The saved initial forecast can be scored later, but rebuilding missing historical forecasts is not prospective evaluation. Frozen-model/persistence comparisons start when their forecasts are actually archived before the target day; do not claim a newly constructed baseline was issued on 17 October.

## Implementation milestones and acceptance

1. **Collector and ledger:** integrate the pinned scraper, live-data schema and immutable storage. Tests cover stale dates, malformed text, partial shops, retries and caching. Pass when identical reruns produce identical committed records without duplication.
2. **Scoring and bootstrap:** import the initial forecast and implement metrics. Hand-counted examples verify precision/recall/Brier, unsupported labels, missing data and combined denominators. Pass when a saved forecast can be scored without changing it or its model.
3. **Versioned refitting:** generalize the existing locked-history forecast/refit code for live inputs without weakening historical verification. Test chronology, training-only support, future-data exclusion and failure recovery. Pass when a daily run creates tomorrow's forecast with complete provenance and preserves original artifact hashes.
4. **Dashboard:** generate a static page with tomorrow's top ten, daily/rolling accuracy, baseline comparisons, coverage, freshness and downloadable history. Verify displayed figures against stored scores and inspect the rendered page.
5. **Automation and deployment:** add scheduled/manual workflows, bounded retries, serialized writes, least-privilege token permissions and publication. Test a manual cloud run and its rerun, then enable schedules. Verify remote state, public dashboard and absence of private assistant notes/credentials.

Run the project suite with `.venv/bin/python -m pytest -q` and the original artifact verification script. Before deployment, require a clean-environment end-to-end run, failure-path tests and a review of the exact public file set.

Each implementation milestone is limited to 90 minutes and three repair cycles. Stop and report partial progress at that limit; do not reset it automatically. No continuous execution loop is activated by this plan. After launch, review at 30 scored days per shop; compare daily refitting with the frozen model and persistence on matching dates. Keep settings fixed during this initial observation period. Any model-search or automatic promotion policy requires a separate prospective protocol.

## Next steps and manual involvement

The immediate next step is local implementation of milestones 1–4, then a reviewable workflow and deployment package. The user has requested a plan, not activated implementation or hosting in this turn.

No manual data collection, Cloudflare account, server, or API key is expected. Public menu access must be verified during implementation.

This checkout currently has no Git remote. Previous publication notes record GitHub authentication as pending; recheck current authentication before asking the user to sign in. If unauthenticated, the user needs to complete GitHub browser/device login. The agent can then use the previously authorized public-repository publication scope to create/push the repository after verification. Resolve an existing intended repository if one exists instead of creating a duplicate.

At the deployment milestone, the user needs to authorize activation of scheduled jobs and public dashboard hosting, which remain outside the existing publication authorization. Prepare and show the final workflow, schedule and published-content scope first. GitHub Actions/Pages settings can be configured programmatically when permitted; ask the user to click settings only if account or organization restrictions prevent this. Use the repository-provided token where sufficient, not a user-supplied personal token. No recurring manual work is intended after launch.
