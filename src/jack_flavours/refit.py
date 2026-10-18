"""Latest-history refit for inference, preserving the evaluation model."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from .features import Dataset
from .final import fit_bundle,digest
from .forecast import build_forecast,forecast


def compare_forecasts(old,new):
    cols=['shop','flavour','probability','rank','top_k']
    merged=old[cols].merge(new[cols],on=['shop','flavour'],how='outer',suffixes=('_old','_new'),indicator=True)
    merged['support_change']=merged['_merge'].map({'both':'shared','left_only':'old_only','right_only':'newly_supported'}).astype(str)
    merged['probability_change_pp']=100*(merged.probability_new-merged.probability_old)
    merged['entered_top10']=merged.top_k_new.eq(True)&~merged.top_k_old.eq(True)
    merged['left_top10']=merged.top_k_old.eq(True)&~merged.top_k_new.eq(True)
    return merged.drop(columns='_merge').sort_values(['shop','rank_new','flavour'],na_position='last')


def refit_latest(config):
    start=time.monotonic()
    def check():
        if time.monotonic()-start>1800:raise TimeoutError('30-minute refit limit reached')
    original=Path('artifacts/final');out=Path('artifacts/latest')
    preserved={str(p):digest(p) for directory in ['artifacts/final','artifacts/validation'] for p in Path(directory).iterdir() if p.is_file()}
    manifest=json.loads((original/'manifest.json').read_text())
    for path,sha in manifest['processed_data'].items():
        if digest(path)!=sha:raise ValueError('Audited history changed: validate source before refitting')
    if digest(original/'model.joblib')!=manifest['outputs']['model.joblib']:raise ValueError('Original model checksum mismatch')
    old=joblib.load(original/'model.joblib')
    data=Dataset.load(Path('data/processed'),old['shops'])
    cutoff=str(data.dates[-1].date())
    if not data.observed[-1].all():raise ValueError('Latest day needs valid snapshots for both shops')
    selection=deepcopy(old['selection'])
    selection['selected']=old['recommended']
    selection['test_period']={'train_end':cutoff}
    selection['operating_point']=old['operating_point']
    if out.exists():
        if not (out/'manifest.json').exists():raise ValueError('Incomplete latest refit exists; inspect before retrying')
        current=json.loads((out/'manifest.json').read_text())
        if current['train_end']!=cutoff or digest(out/'model.joblib')!=current['outputs']['model.joblib']:
            raise ValueError('Latest model differs; preserve it before a new versioned refit')
        bundle=joblib.load(out/'model.joblib')
    else:
        out.mkdir(parents=True)
        bundle=fit_bundle(data,selection,config,check)
        bundle.update(recommended=old['recommended'],operating_point=old['operating_point'],training_role='latest_history_refit',evaluation_reference='artifacts/final',original_selection_sha256=digest('artifacts/validation/selection.json'))
        joblib.dump(bundle,out/'model.joblib')
        newmanifest={'train_end':cutoff,'training_role':'latest_history_refit','not_independently_evaluated':True,'source_commit':bundle['source_commit'],'original_selection_sha256':bundle['original_selection_sha256'],'scientific_inputs':{p:digest(p) for p in ['src/jack_flavours/features.py','src/jack_flavours/models.py','src/jack_flavours/refit.py','src/jack_flavours/final.py','configs/experiment.json','uv.lock']},'processed_data':manifest['processed_data'],'outputs':{'model.joblib':digest(out/'model.joblib')},'created_utc':datetime.now(timezone.utc).isoformat()}
        (out/'manifest.json').write_text(json.dumps(newmanifest,indent=2)+'\n')
    oldframe,_=build_forecast(old,data)
    newframe,meta=build_forecast(bundle,data)
    comparison=compare_forecasts(oldframe,newframe)
    comparison.to_csv(out/'forecast-comparison.csv',index=False)
    summary={'old_train_end':old['train_end'],'new_train_end':bundle['train_end'],'target_date':meta['target_date'],'old_training_forecasts':old['train_forecasts'],'new_training_forecasts':bundle['train_forecasts'],'old_supported_flavours':len(old['labels']),'new_supported_flavours':len(bundle['labels']),'shops':{}}
    for shop,df in comparison.groupby('shop'):
        summary['shops'][shop]={'retained_top10':int((df.top_k_old.eq(True)&df.top_k_new.eq(True)).sum()),'entrants':df.loc[df.entered_top10,'flavour'].tolist(),'departures':df.loc[df.left_top10,'flavour'].tolist(),'mean_absolute_probability_change_pp_shared':float(df.loc[df.support_change=='shared','probability_change_pp'].abs().mean())}
    result=forecast(config,model_dir=out)
    summary['forecast_files']={k:result[k] for k in ['forecast','top_forecast','metadata']}
    (out/'comparison-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for p,sha in preserved.items():assert digest(p)==sha,p
    lines=['# Tomorrow forecast: July model versus latest-history refit','',f"Target: **{meta['target_date']}**, using the same issue-day observations. Old training cutoff **{old['train_end']}**; new cutoff **{bundle['train_end']}**. Added the last three months to the earlier history, not a three-month-only window.",'',f"Eligible training shop/day examples: {old['train_forecasts']} → {bundle['train_forecasts']}. Supported flavour labels: {len(old['labels'])} → {len(bundle['labels'])}.",'', 'Same L2 logit C=0.01, original features, minimum five training appearances, threshold 0.2 and top ten per shop. No novelty features or hyperparameter tuning. Updated data can change fitted weights, scalers and label support. New labels below refer to newly supported model labels, not necessarily genuinely new recipes.','']
    for shop,df in comparison.groupby('shop'):
        lines += [f'## {shop}','', '| New rank | Flavour | July model probability | Updated probability | Change | Old rank |','|---|---|---:|---:|---:|---:|']
        for row in df[df.top_k_new.eq(True)].itertuples():
            oldp='Not supported' if pd.isna(row.probability_old) else f'{row.probability_old:.1%}'
            delta='Newly supported' if pd.isna(row.probability_change_pp) else f'{row.probability_change_pp:+.1f} pp'
            oldrank='—' if pd.isna(row.rank_old) else str(int(row.rank_old))
            lines.append(f'| {int(row.rank_new)} | {row.flavour} | {oldp} | {row.probability_new:.1%} | {delta} | {oldrank} |')
        info=summary['shops'][shop]
        lines+=['',f"Top-ten overlap: {info['retained_top10']}/10. Entries: {', '.join(info['entrants']) or 'none'}. Exits: {', '.join(info['departures']) or 'none'}.",'']
    lines+=['## Interpretation and usage','','These are changed forecasts, not evidence that the updated model is more accurate: tomorrow’s menu is not yet observed. The latest model has trained on the previously evaluated July–October period; the original out-of-sample results belong only to the preserved July model. No historical test is rerun or claimed for this refit. Probabilities remain imperfectly calibrated.','', 'Run `.venv/bin/python -m jack_flavours forecast` for the latest model, or add `--model evaluation` to use the preserved July model. To manually create/reuse this latest audited-history refit and regenerate the comparison, run `.venv/bin/python scripts/refit_latest.py`. No source download occurs. Latest outputs live under artifacts/forecasts/latest; old forecast files remain unchanged.','', 'Full outer-joined comparison: artifacts/latest/forecast-comparison.csv. Unsupported old labels have missing old probabilities, not zero. The original validation and final artifact hashes were verified unchanged.']
    Path('reports/latest-refit-comparison.md').write_text('\n'.join(lines)+'\n')
    return summary
