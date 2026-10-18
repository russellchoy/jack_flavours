# Tomorrow forecast: July model versus latest-history refit

Target: **2026-10-18**, using the same issue-day observations. Old training cutoff **2026-07-19**; new cutoff **2026-10-17**. Added the last three months to the earlier history, not a three-month-only window.

Eligible training shop/day examples: 980 → 1158. Supported flavour labels: 461 → 526.

Same L2 logit C=0.01, original features, minimum five training appearances, threshold 0.2 and top ten per shop. No novelty features or hyperparameter tuning. Updated data can change fitted weights, scalers and label support. New labels below refer to newly supported model labels, not necessarily genuinely new recipes.

## all_saints

| New rank | Flavour | July model probability | Updated probability | Change | Old rank |
|---|---|---:|---:|---:|---:|
| 1 | pink guava & mezcal | Not supported | 99.6% | Newly supported | — |
| 2 | victoria sponge | Not supported | 98.8% | Newly supported | — |
| 3 | lemon yoghurt | 99.9% | 98.3% | -1.6 pp | 1 |
| 4 | mascarpone vanilla | Not supported | 98.0% | Newly supported | — |
| 5 | dark chocolate & sea salt | 95.6% | 96.0% | +0.4 pp | 2 |
| 6 | coconut ube | 89.0% | 92.8% | +3.8 pp | 3 |
| 7 | alphonso mango sorbet | 87.4% | 92.2% | +4.8 pp | 4 |
| 8 | hong kong milk tea | 84.4% | 87.7% | +3.3 pp | 5 |
| 9 | grilled white chocolate | 80.1% | 84.6% | +4.5 pp | 6 |
| 10 | girton apple sorbet | 68.8% | 70.5% | +1.7 pp | 7 |

Top-ten overlap: 7/10. Entries: pink guava & mezcal, victoria sponge, mascarpone vanilla. Exits: blackcurrants & cream, tiramisu, coconut & ube.

## benet_street

| New rank | Flavour | July model probability | Updated probability | Change | Old rank |
|---|---|---:|---:|---:|---:|
| 1 | dark chocolate & sea salt | 97.5% | 97.9% | +0.4 pp | 1 |
| 2 | strawberries & cream | 89.9% | 90.0% | +0.1 pp | 2 |
| 3 | salted caramel | 56.0% | 70.0% | +13.9 pp | 5 |
| 4 | house vanilla | 74.5% | 66.0% | -8.6 pp | 3 |
| 5 | king’s college lavender | 61.5% | 65.1% | +3.6 pp | 4 |
| 6 | apricot sorbet | 55.7% | 63.3% | +7.6 pp | 6 |
| 7 | gin & tonic sorbet | 44.9% | 56.8% | +11.9 pp | 8 |
| 8 | honeycomb | 52.4% | 55.6% | +3.2 pp | 7 |
| 9 | coconut sugar | 36.4% | 46.4% | +10.0 pp | 9 |
| 10 | chocolate | 24.9% | 26.5% | +1.6 pp | 10 |

Top-ten overlap: 10/10. Entries: none. Exits: none.

## Interpretation and usage

These are changed forecasts, not evidence that the updated model is more accurate: tomorrow’s menu is not yet observed. The latest model has trained on the previously evaluated July–October period; the original out-of-sample results belong only to the preserved July model. No historical test is rerun or claimed for this refit. Probabilities remain imperfectly calibrated.

Run `.venv/bin/python -m jack_flavours forecast` for the latest model, or add `--model evaluation` to use the preserved July model. To manually create/reuse this latest audited-history refit and regenerate the comparison, run `.venv/bin/python scripts/refit_latest.py`. No source download occurs. Latest outputs live under artifacts/forecasts/latest; old forecast files remain unchanged.

Full outer-joined comparison: artifacts/latest/forecast-comparison.csv. Unsupported old labels have missing old probabilities, not zero. The original validation and final artifact hashes were verified unchanged.
