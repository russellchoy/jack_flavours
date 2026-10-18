"""Verify saved experiment integrity and reproduce predictions without retraining."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from jack_flavours.features import Dataset
from jack_flavours.final import digest,predict_bundle
from jack_flavours.forecast import build_forecast

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--portable', action='store_true', help='Allow four float32 ULP-scale absolute rounding units across CPU/BLAS platforms; checksums remain exact')
args=parser.parse_args()
tolerance=4*np.finfo(np.float32).eps if args.portable else 1e-12
max_error=0.0
root=Path('artifacts/final')
manifest=json.loads((root/'manifest.json').read_text())
for name,sha in manifest['outputs'].items():
    assert digest(root/name)==sha,name
for name,sha in manifest['processed_data'].items():
    assert digest(name)==sha,name
bundle=joblib.load(root/'model.joblib')
data=Dataset.load(Path('data/processed'),bundle['shops'])
period=bundle['selection']['test_period']
rows=np.flatnonzero(data.mask(period['start'],period['end']))
saved=np.load(root/'predictions.npz',allow_pickle=False)
replayed,indices=predict_bundle(bundle,data,rows)
for model,p in replayed.items():
    max_error=max(max_error,float(np.max(np.abs(p-saved[model]))))
    np.testing.assert_allclose(p,saved[model],rtol=0,atol=tolerance)
np.testing.assert_array_equal(data.y[rows][:,indices],saved['y'])
np.testing.assert_array_equal(data.rows.iloc[rows].target_date.to_numpy().astype('datetime64[D]').astype(str),saved['dates'])
np.testing.assert_array_equal(data.rows.iloc[rows].shop.to_numpy().astype(str),saved['shops'])
forecast,meta=build_forecast(bundle,data)
base=Path('artifacts/forecasts')/('forecast-'+meta['target_date'])
if not base.with_suffix('.json').exists():
    # The original forecasts may be filed under the user-named comparison folder.
    base=Path('artifacts/forecasts/minus_L3M')/('forecast-'+meta['target_date'])
record=json.loads(base.with_suffix('.json').read_text())
assert digest(base.with_suffix('.csv'))==record['csv_sha256']
assert digest(Path(str(base)+'-top.csv'))==record['top_csv_sha256']
actual=pd.read_csv(base.with_suffix('.csv'))
np.testing.assert_allclose(forecast.probability,actual.probability,rtol=0,atol=tolerance if args.portable else 1e-15)
np.testing.assert_array_equal(forecast.top_k,actual.top_k)
np.testing.assert_array_equal(forecast.flavour,actual.flavour)
assert not actual.duplicated(['shop','flavour']).any()
assert len(actual)==len(bundle['labels'])*len(meta['forecast_shops'])
assert actual.probability.between(0,1).all()
assert actual.top_k.sum()==meta['operating_point']['k']*len(meta['forecast_shops'])
report=json.loads((root/'metrics.json').read_text())
for model in report['metrics']:
    m=report['metrics'][model]
    assert sum(x['count'] for x in m['calibration'])==m['label_forecast_pairs']
    assert m['forecasts']==len(rows)
print(json.dumps({'verified':True,'portable':args.portable,'absolute_tolerance':float(tolerance),'maximum_replay_error':max_error,'saved_model_reproduces_all_test_predictions':True,'heldout_forecasts':len(rows),'model_labels':len(bundle['labels']),'forecast_rows':len(actual),'target_date':meta['target_date'],'checksums_verified':len(manifest['outputs'])},indent=2))
