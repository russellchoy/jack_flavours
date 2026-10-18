"""Forecast the next snapshot from a trusted local saved model, without refitting."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .features import Dataset
from .final import digest,predict_bundle


def build_forecast(bundle,data,issue_date=None):
    issue=pd.Timestamp(issue_date) if issue_date else data.dates[-1]
    if issue not in data.dates:raise ValueError('Issue date must exist in the supplied snapshot history')
    if issue<pd.Timestamp(bundle['train_end']):raise ValueError('Issue date precedes model training cutoff')
    stop=int(data.dates.get_loc(issue))+1
    valid=data.observed[stop-1]
    if not valid.any():raise ValueError('No valid issue-date snapshot; refusing a stale forecast')
    # Discard ALL observations after issue date, even if supplied historical data contains them.
    history=Dataset(pd.date_range(data.dates[0],issue+pd.Timedelta(days=1)),data.shops,data.names,np.concatenate([data.observed[:stop].copy(),np.zeros((1,len(data.shops)),bool)]),np.concatenate([data.presence[:stop].copy(),np.zeros((1,len(data.shops),len(data.names)),np.uint8)]))
    rows=stop*len(data.shops)+np.flatnonzero(valid)
    p,_=predict_bundle(bundle,history,rows)
    selected=bundle['recommended'];prob=p[selected];point=bundle['operating_point']
    output=[]
    for i,s in enumerate(np.flatnonzero(valid)):
        order=np.argsort(-prob[i],kind='stable')
        for rank,j in enumerate(order,1):
            output.append({'issue_date':str(issue.date()),'target_date':str((issue+pd.Timedelta(days=1)).date()),'shop':data.shops[s],'flavour':bundle['labels'][j],'probability':float(prob[i,j]),'rank':rank,'top_k':rank<=point['k'],'above_threshold':bool(prob[i,j]>=point['threshold']),'model':selected,'model_train_end':bundle['train_end']})
    frame=pd.DataFrame(output)
    metadata={'issue_date':str(issue.date()),'target_date':str((issue+pd.Timedelta(days=1)).date()),'model':selected,'model_train_end':bundle['train_end'],'supported_flavours':len(bundle['labels']),'forecast_shops':[data.shops[s] for s in np.flatnonzero(valid)],'unavailable_shops':[data.shops[s] for s in np.flatnonzero(~valid)],'operating_point':point,'scope':'Next snapshot presence, not all-day stock. New/unsupported flavours cannot be predicted. Probability calibration remains imperfect. Training cutoff is reported explicitly; original evaluation scores do not validate a later refit.'}
    return frame,metadata


def forecast(config,issue_date=None,out=None,model_dir=None):
    model_dir=Path(model_dir) if model_dir else (Path('artifacts/latest') if Path('artifacts/latest/manifest.json').exists() else Path('artifacts/final'))
    out=Path(out) if out is not None else (Path('artifacts/forecasts/latest') if model_dir.name=='latest' else Path('artifacts/forecasts'))
    manifest=json.loads((model_dir/'manifest.json').read_text())
    if digest(model_dir/'model.joblib')!=manifest['outputs']['model.joblib']:raise ValueError('Model checksum mismatch')
    # The command only loads the locally generated trusted bundle after checksum verification.
    for p in ['src/jack_flavours/features.py','src/jack_flavours/models.py']:
        if digest(p)!=manifest['scientific_inputs'][p]:raise ValueError('Model feature/estimator implementation changed')
    for p,sha in manifest['processed_data'].items():
        if digest(p)!=sha:raise ValueError('History changed; audit the new source before using this pinned experiment')
    bundle=joblib.load(model_dir/'model.joblib')
    data=Dataset.load(Path('data/processed'),bundle['shops'])
    frame,metadata=build_forecast(bundle,data,issue_date)
    metadata.update({'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':manifest['outputs']['model.joblib'],'source_commit':bundle['source_commit'],'history_hashes':manifest['processed_data'],'model_directory':str(model_dir),'training_role':bundle.get('training_role','heldout_evaluation_model')})
    out.mkdir(parents=True,exist_ok=True)
    stem='forecast-'+metadata['target_date']
    full=out/(stem+'.csv');top=out/(stem+'-top.csv');meta=out/(stem+'.json')
    frame.to_csv(full,index=False);frame[frame.top_k].to_csv(top,index=False)
    metadata['csv_sha256']=digest(full);metadata['top_csv_sha256']=digest(top)
    meta.write_text(json.dumps(metadata,indent=2)+'\n')
    return {'forecast':str(full),'top_forecast':str(top),'metadata':str(meta),**metadata}
