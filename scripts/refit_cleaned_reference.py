from pathlib import Path
from copy import deepcopy
import json,time,joblib,pandas as pd
from jack_flavours.daily import trusted_latest
from jack_flavours.live_cleaning import canonical_title_v5
from jack_flavours.features import Dataset
from jack_flavours.final import fit_bundle,digest
out=Path('artifacts/production-v5');out.mkdir(exist_ok=True)
assert not (out/'model.joblib').exists()
old=trusted_latest();selection=deepcopy(old['selection']);selection.update(selected='logit',logistic_C=.01,min_support=5,test_period={'train_end':'2026-10-17'})
a=pd.read_parquet('data/processed/appearances.parquet');a['canonical_name']=a.raw_name.map(canonical_title_v5)
s=pd.read_parquet('data/processed/snapshots.parquet');data=Dataset.from_tables(s,a,old['shops'])
started=time.monotonic()
def check():
 if time.monotonic()-started>1800:raise TimeoutError('30-minute v5 fit cap')
bundle=fit_bundle(data,selection,json.loads(Path('configs/experiment.json').read_text()),check)
bundle.update(recommended='logit',operating_point=old['operating_point'],normalization_version='v5',training_role='frozen_v5_reference')
joblib.dump(bundle,out/'model.joblib')
manifest={'normalization':'v5','train_end':'2026-10-17','source':'Fixed settings with all 83 user-approved candidate groups and house vanilla alias; fitted 18 October','model_sha256':digest(out/'model.joblib'),'implementation_sha256':{p:digest(p) for p in ['src/jack_flavours/cleaning.py','src/jack_flavours/live_cleaning.py','src/jack_flavours/approved_aliases_v5.json','src/jack_flavours/features.py','src/jack_flavours/models.py']},'processed_data':json.loads(Path('artifacts/latest/manifest.json').read_text())['processed_data']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('v5 reference labels:',len(bundle['labels']))
