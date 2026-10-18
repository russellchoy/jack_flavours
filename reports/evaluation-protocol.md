# Frozen evaluation protocol

```json
{
  "splits": {
    "folds": [
      {
        "name": "validation_1",
        "train_end": "2026-04-20",
        "start": "2026-04-21",
        "end": "2026-05-20"
      },
      {
        "name": "validation_2",
        "train_end": "2026-05-20",
        "start": "2026-05-21",
        "end": "2026-06-19"
      },
      {
        "name": "validation_3",
        "train_end": "2026-06-19",
        "start": "2026-06-20",
        "end": "2026-07-19"
      }
    ],
    "test": {
      "train_end": "2026-07-19",
      "start": "2026-07-20",
      "end": "2026-10-17"
    },
    "policy": "Split on target date; both shops share dates. Eligible iff target and previous calendar day are valid. Test outcomes remain unused until selection is frozen."
  },
  "training_validation_inventory": [
    {
      "name": "validation_1",
      "train_end": "2026-04-20",
      "start": "2026-04-21",
      "end": "2026-05-20",
      "train_forecasts": 804,
      "validation_forecasts": 60,
      "eligible_training_flavours": 410
    },
    {
      "name": "validation_2",
      "train_end": "2026-05-20",
      "start": "2026-05-21",
      "end": "2026-06-19",
      "train_forecasts": 864,
      "validation_forecasts": 60,
      "eligible_training_flavours": 434
    },
    {
      "name": "validation_3",
      "train_end": "2026-06-19",
      "start": "2026-06-20",
      "end": "2026-07-19",
      "train_forecasts": 924,
      "validation_forecasts": 56,
      "eligible_training_flavours": 449
    }
  ]
}
```

Support: at least five positive eligible training targets per flavour. All shops remain on the same side of each split. No test flavour frequencies or scores are reported here.

Features use only issue-day and older observations. Rates use observed counts in calendar windows; unknown observations are omitted from denominators. Streaks break at gaps. Previous-day validity is required for all compared predictors. Initial training-only transforms and vocabulary are refit per fold.
