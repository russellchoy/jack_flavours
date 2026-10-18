# Free daily deployment

## Services and cost controls

Use a public GitHub repository and standard `ubuntu-latest` runners; the workflow skips private repositories. No paid runners, Cloudflare Workers paid plan, database, LLM API or custom domain is needed. Select Cloudflare Pages **Free** and retain the provided `pages.dev` address. This project must not enable paid services or automatic upgrades.

GitHub schedules cover both GMT/BST UTC offsets, and the application accepts only 11:xx, 13:xx and 16:xx Europe/London. Each scheduled trigger is at minute 37. There are at most three collection attempts per local date, including manual attempts, and one forecast issuance per day. Missed/delayed jobs do not create retrospective forecasts. Source freshness is validated independently at each shop. Any failed run retains completed scoring; a new adaptive forecast requires a successful refit.

A data update generates at most one Pages build per pipeline attempt (normally one per day). This is far below Pages Free's 500 builds/month under this schedule. Disable preview builds, which are unnecessary here. Confirm allowances in the service account before changing this schedule.

## GitHub

1. Publish the verified source on `main`, with local assistant notes and credentials excluded.
2. Create `daily-data` containing validated initial state and `.gitignore` excluding `models/` and temporary files.
3. Upload each locally issued model as release asset `YYYY-MM-DD.joblib` under tag `daily-model-YYYY-MM-DD`; verify its hash against `issued/YYYY-MM-DD.json` before publishing the data commit.
4. Run **Verify model and daily pipeline**, then manually run **Daily menu forecast**. Inspect Actions results, saved data and any release assets.
5. The committed cron schedule then runs automatically. GitHub schedules can be delayed or disabled after prolonged repository inactivity; the dashboard shows freshness. GitHub's own workflow notification settings can notify the account owner of failures.

The workflow writes data with the repository-provided token. No personal access token is required for daily operation. An interrupted model upload or checksum conflict fails closed; do not overwrite the asset to make it pass.

## Cloudflare Pages account connection

In Cloudflare **Workers & Pages**, choose **Create application → Pages → Connect to Git** (labels may vary). Select the intended GitHub repository and authorize access only to that repository when possible.

- Project name: `jack-flavours` if available.
- Production branch: `daily-data`.
- Framework preset: None.
- Build command: empty.
- Build output directory: `site`.
- Root directory: repository root.
- Plan: Free.
- Preview branches: disabled.

The Pages project reads only generated static HTML and JSON. It does not execute the model or need any secrets. Verify that `index.html` and `data.json` load on the returned `pages.dev` URL. Check that a later `daily-data` commit causes a new deployment; a dashboard deployment failure does not erase model/scoring records.

The user may need to sign into Cloudflare and approve its GitHub installation. Do not paste credentials or tokens into chat. If an integration approval screen requires the account owner's consent, complete it in the browser.

## Operational checks

- Inspect the displayed target date before treating a forecast as current.
- A missing shop remains unavailable; no stale menu is counted as today's outcome.
- Scores are prospective only when the saved creation timestamp is on the preceding London calendar day.
- Raw historical scores and per-model probability scores use different support sets; compare probability quality via the saved shared-label comparison.
- Manual recovery must preserve source timestamps. Backfilling missing forecasts is forbidden as prospective evidence.
- Review storage and model retention after 90 days. Review predictive performance after 30 scored days per shop. No automatic hyperparameter tuning or promotion is enabled.
