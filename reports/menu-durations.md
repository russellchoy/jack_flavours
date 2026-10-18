# Menu stay and return durations

Descriptive history: 4 October 2024–17 October 2026, pinned source fba48c936c059625f399353e5afa8b071fc51457. No model fitting or retuning.

| Shop | On-menu mean | On-menu median | Completed stays | Off-menu mean | Off-menu median | Completed return gaps |
|---|---:|---:|---:|---:|---:|---:|
| both_shops | 2.77 | 2 | 3887 | 11.09 | 6 | 1401 |
| all_saints | 3.87 | 3 | 1041 | 10.28 | 4 | 381 |
| benet_street | 2.37 | 2 | 2846 | 11.40 | 6 | 1020 |

Each observation is a flavour at one shop. Means are weighted by completed spells, not equally by flavour. A completed stay begins after an observed absent day and ends at the next observed absent day. A completed return gap begins after a present day and ends at the next present day. Daily snapshots are not continuous stock observations.

Only uninterrupted, valid daily observations with both transitions confirmed enter these means. Missing, suspect or failed snapshots censor a spell; the history boundary also censors it. Spells already in progress at the start of a valid segment are excluded. A one-day absence means absent in one daily snapshot before returning the following day.

These are conditional completed-spell averages, NOT unconditional expected wait times. Long stays/gaps are more likely to cross a missing day or the history endpoint and be excluded. Flavours that never return do not enter the completed-return mean. Conservative name matching also treats unmerged spelling/name changes as different flavours. Do not infer a guaranteed return schedule from these figures.

A separate sensitivity summary includes next observed appearances across unknown days: that is elapsed time until next observed return, not proven continuous absence, because unobserved intermediate returns are possible.

```json
{
  "both_shops": {
    "on_menu": {
      "completed_spells": 3887,
      "mean_days": 2.7707743761255466,
      "median_days": 2.0,
      "p25_days": 1.0,
      "p75_days": 3.0,
      "p90_days": 5.0,
      "one_day_pct": 37.483920761512735,
      "incomplete_known_start_spells": 492,
      "censored_at_gap": 464,
      "censored_at_history_end": 28
    },
    "off_menu_until_return": {
      "completed_spells": 1401,
      "mean_days": 11.093504639543184,
      "median_days": 6.0,
      "p25_days": 2.0,
      "p75_days": 14.0,
      "p90_days": 29.0,
      "one_day_pct": 17.273376159885796,
      "incomplete_known_start_spells": 2964,
      "censored_at_gap": 2578,
      "censored_at_history_end": 386
    }
  },
  "all_saints": {
    "on_menu": {
      "completed_spells": 1041,
      "mean_days": 3.8683957732949086,
      "median_days": 3.0,
      "p25_days": 2.0,
      "p75_days": 5.0,
      "p90_days": 8.0,
      "one_day_pct": 22.382324687800192,
      "incomplete_known_start_spells": 199,
      "censored_at_gap": 184,
      "censored_at_history_end": 15
    },
    "off_menu_until_return": {
      "completed_spells": 381,
      "mean_days": 10.280839895013123,
      "median_days": 4.0,
      "p25_days": 2.0,
      "p75_days": 12.0,
      "p90_days": 28.0,
      "one_day_pct": 21.784776902887142,
      "incomplete_known_start_spells": 848,
      "censored_at_gap": 768,
      "censored_at_history_end": 80
    }
  },
  "benet_street": {
    "on_menu": {
      "completed_spells": 2846,
      "mean_days": 2.369290231904427,
      "median_days": 2.0,
      "p25_days": 1.0,
      "p75_days": 3.0,
      "p90_days": 5.0,
      "one_day_pct": 43.00773014757554,
      "incomplete_known_start_spells": 293,
      "censored_at_gap": 280,
      "censored_at_history_end": 13
    },
    "off_menu_until_return": {
      "completed_spells": 1020,
      "mean_days": 11.397058823529411,
      "median_days": 6.0,
      "p25_days": 2.0,
      "p75_days": 14.0,
      "p90_days": 29.0,
      "one_day_pct": 15.588235294117647,
      "incomplete_known_start_spells": 2116,
      "censored_at_gap": 1810,
      "censored_at_history_end": 306
    }
  },
  "observed_returns_including_intervening_unknown_days": {
    "both_shops": {
      "count": 2589,
      "mean_calendar_days": 51.31711085361143,
      "median_calendar_days": 15.0,
      "with_unknown_days": 1188
    },
    "all_saints": {
      "count": 691,
      "mean_calendar_days": 57.178002894356005,
      "median_calendar_days": 14.0,
      "with_unknown_days": 310
    },
    "benet_street": {
      "count": 1898,
      "mean_calendar_days": 49.183350895679666,
      "median_calendar_days": 15.0,
      "with_unknown_days": 878
    }
  }
}
```

Reproduce with `.venv/bin/python scripts/menu_durations.py`. Three hand-calculated synthetic histories verify complete spells, missing-day censoring and unknown initial starts. Full spell-level records remain local in artifacts/menu-durations/spells.csv.

## Wider observed-return interpretation

Across all confirmed departures, including subsequent unknown dates, the next recorded return averages 51.3 calendar days (median 15): All Saints 57.2 (median 14), Bene't Street 49.2 (median 15). These figures are conditional on a later recorded return, and do not prove continuous absence across gaps. Of 4,365 confirmed departures, 2,589 have a later recorded return; 1,776 (40.7%) do not before history ends. They may be seasonal, discontinued, renamed or simply not yet observed again. The strict uninterrupted-gap mean of 11.1 days is a shorter-gap subset, not a reliable unconditional expected waiting time.
