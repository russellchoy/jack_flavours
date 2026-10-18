"""One frozen held-out evaluation, persisted model, and paired uncertainty."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .features import Dataset, FEATURES
from .models import fit_binary
from .evaluation import metrics
from .experiment import fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_deadline(session):
    if datetime.now(timezone.utc)>=datetime.fromisoformat(session['deadline_utc']):
        raise TimeoutError('Authorized continuation deadline reached')


def verify_frozen(selection,session):
    if digest('artifacts/validation/selection.json')!=session['validation_selection_sha256']:
        raise ValueError('Frozen validation selection changed')
    # CLI wiring may change; scientific implementations and experiment settings may not.
    for path,sha in selection['files'].items():
        if path.endswith('__main__.py'):continue
        if digest(path)!=sha:raise ValueError(f'Frozen scientific input changed: {path}')


def fit_bundle(data,selection,config,check=lambda:None):
    cutoff=selection['test_period']['train_end']
    train=data.mask(end=cutoff); indices=data.vocabulary(train,selection['min_support'])
    if not train.any() or not len(indices):raise ValueError('No eligible training data')
    models=[]; rates=[]
    shops=data.rows.shop.to_numpy()
    y=data.y[train][:,indices]
    for shop in data.shops:
        mask=train & (shops==shop)
        rates.append(data.y[mask][:,indices].mean(axis=0) if mask.any() else y.mean(axis=0))
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for j,index in enumerate(indices):
            check()
            if selection['selected'] in ('logit','boosting'):
                x=data.features(index)
                model=fit_binary(x[train],data.y[train,index],selection['selected'],config,selection['logistic_C'])
                models.append(model)
            if (j+1)%100==0:print(f'Final refit: {j+1}/{len(indices)} flavours',flush=True)
    return {'schema_version':1,'candidate':selection['selected'],'labels':[data.names[i] for i in indices],'shops':data.shops,'feature_names':FEATURES,'models':models,'frequency':np.array(rates),'train_end':cutoff,'source_commit':selection['source_commit'],'train_forecasts':int(train.sum()),'train_support':y.sum(axis=0),'selection':selection,'history_start':str(data.dates[0].date())}


def predict_bundle(bundle,data,rows,check=lambda:None):
    if bundle['shops']!=data.shops or bundle['feature_names']!=FEATURES:
        raise ValueError('Model schema or shop order mismatch')
    if data.rows.iloc[rows].target_date.min()<=pd.Timestamp(bundle['train_end']):
        raise ValueError('Prediction target must be after model training cutoff')
    lookup={name:i for i,name in enumerate(data.names)}
    if not set(bundle['labels'])<=set(lookup):raise ValueError('Model label missing from history')
    indices=np.array([lookup[name] for name in bundle['labels']])
    rows=np.asarray(rows); nshops=len(data.shops)
    if (rows<nshops).any() or not data.observed.ravel()[rows-nshops].all():
        raise ValueError('Every forecast needs a valid previous-calendar-day snapshot')
    p={'frequency':bundle['frequency'][rows%nshops],'persistence':data.y[rows-nshops][:,indices].astype(float)}
    if bundle['candidate'] not in p:
        p[bundle['candidate']]=np.empty((len(rows),len(indices)))
        with threadpool_limits(limits=1):
            for j,index in enumerate(indices):
                check();x=data.features(index)
                p[bundle['candidate']][:,j]=bundle['models'][j].predict_proba(x[rows])[:,1]
    return p,indices


def sample_calendar_blocks(dates,start,end,block_days,rng):
    """Moving blocks of calendar dates; keep all available shop rows together."""
    days=pd.date_range(start,end)
    if block_days<1 or block_days>len(days):raise ValueError('Invalid block length')
    starts=rng.integers(0,len(days)-block_days+1,size=int(np.ceil(len(days)/block_days)))
    sampled=np.concatenate([np.arange(i,i+block_days) for i in starts])[:len(days)]
    values=np.asarray(dates,dtype='datetime64[D]')
    positions=[np.flatnonzero(values==day.to_datetime64().astype('datetime64[D]')) for day in days]
    return np.concatenate([positions[i] for i in sampled])


def bootstrap_differences(y,p,dates,candidate,start,end,block_days=7,repeats=200,seed=42,check=lambda:None):
    rng=np.random.default_rng(seed)
    baselines=[b for b in ['frequency','persistence'] if b!=candidate]
    samples={b:{'ap_gain':[],'brier_reduction':[]} for b in baselines}
    for _ in range(repeats):
        check();rows=sample_calendar_blocks(dates,start,end,block_days,rng)
        if not len(rows) or not y[rows].sum():continue
        truth=y[rows].ravel();cp=p[candidate][rows].ravel()
        cap=average_precision_score(truth,cp);cb=np.mean((truth-cp)**2)
        for b in baselines:
            bp=p[b][rows].ravel()
            samples[b]['ap_gain'].append(float(cap-average_precision_score(truth,bp)))
            samples[b]['brier_reduction'].append(float(np.mean((truth-bp)**2)-cb))
    output={}
    for b,values in samples.items():
        output[b]={}
        for metric,draws in values.items():
            if len(draws)<max(2,repeats//2):raise ValueError('Too few informative bootstrap draws')
            output[b][metric]={'ci95':np.quantile(draws,[.025,.975]).tolist(),'replicates':len(draws),'positive_means_candidate_better':True}
    return output


def final_decision(results,candidate,fallback,intervals):
    gate=all(results[candidate]['micro_ap']>results[b]['micro_ap'] and results[candidate]['brier']<=results[b]['brier'] for b in ['frequency','persistence'] if b!=candidate)
    supported=gate and all(v['ap_gain']['ci95'][0]>0 and v['brier_reduction']['ci95'][0]>0 for v in intervals.values())
    if candidate in ['frequency','persistence']:
        return {'recommended':candidate,'test_gate_passed':None,'uncertainty':'baseline selected in validation'}
    return {'recommended':candidate if gate else fallback,'test_gate_passed':gate,'uncertainty':'improvement supported by paired intervals' if supported else 'inconclusive improvement; do not claim statistically established superiority'}


def evaluate_final(config):
    session=json.loads(Path('configs/continuation.json').read_text())
    selection=json.loads(Path('artifacts/validation/selection.json').read_text())
    verify_frozen(selection,session)
    out=Path('artifacts/final')
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for name,sha in manifest['outputs'].items():
            if digest(out/name)!=sha:raise ValueError('Saved final output changed: '+name)
        print('Returning saved final evaluation; no refitting or rescoring.',flush=True)
        return json.loads((out/'metrics.json').read_text())['decision']
    check_deadline(session)
    if out.exists():raise ValueError('Partial final run exists: inspect and resume saved artifacts; never overwrite or retune')
    # Verify source and derived provenance before opening the held-out outcomes.
    sha=subprocess.check_output(['git','-C','data/source','rev-parse','HEAD'],text=True).strip()
    if sha!=selection['source_commit'] or subprocess.check_output(['git','-C','data/source','status','--porcelain'],text=True).strip():raise ValueError('Source checkout drift')
    audit=json.loads(Path('data/processed/audit.json').read_text())
    if audit['source_commit']!=sha:raise ValueError('Processed source revision mismatch')
    data=Dataset.load(Path('data/processed'),config['shops'])
    out.mkdir(parents=True,exist_ok=False)
    manifest={'source_commit':sha,'selection_sha256':digest('artifacts/validation/selection.json'),'scientific_inputs':fingerprint(),'new_stage_files':{str(p):digest(p) for p in [Path('src/jack_flavours/final.py'),Path('configs/continuation.json')]},'processed_data':{str(p):digest(p) for p in Path('data/processed').glob('*.parquet')},'started_utc':datetime.now(timezone.utc).isoformat(),'availability_assumption':config['cutoff']}
    (out/'started.json').write_text(json.dumps(manifest,indent=2)+'\n')
    bundle=fit_bundle(data,selection,config,lambda:check_deadline(session))
    joblib.dump(bundle,out/'model.joblib')
    # Weights, transforms and label vocabulary are now fixed before test prediction/scoring.
    test=selection['test_period'];rows=np.flatnonzero(data.mask(test['start'],test['end']))
    if not len(rows):raise ValueError('No eligible test forecasts')
    predictions,indices=predict_bundle(bundle,data,rows,lambda:check_deadline(session))
    y=data.y[rows][:,indices];total=data.y[rows].sum(axis=1)
    dates=data.rows.iloc[rows].target_date.to_numpy().astype('datetime64[D]').astype(str)
    shops=data.rows.iloc[rows].shop.to_numpy().astype(str)
    val=json.loads(Path('artifacts/validation/metrics.json').read_text())
    results={};per_shop={}
    for model,p in predictions.items():
        point=val[model]['operating_point']
        results[model]=metrics(y,p,total,**point)
        per_shop[model]={s:metrics(y[shops==s],p[shops==s],total[shops==s],**point) for s in bundle['shops'] if (shops==s).any()}
    np.savez_compressed(out/'predictions.npz',y=y,total=total,dates=dates,shops=shops,labels=np.array(bundle['labels']),**predictions)
    intervals=bootstrap_differences(y,predictions,dates,bundle['candidate'],test['start'],test['end'],config['bootstrap_block_days'],config['bootstrap_replicates'],config['seed'],lambda:check_deadline(session))
    decision=final_decision(results,bundle['candidate'],selection['baseline_fallback'],intervals)
    bundle['recommended']=decision['recommended']
    bundle['operating_point']=val[decision['recommended']]['operating_point']
    joblib.dump(bundle,out/'model.joblib')
    all_test=(data.rows.target_date>=pd.Timestamp(test['start']))&(data.rows.target_date<=pd.Timestamp(test['end']))
    target_valid=data.observed.ravel()
    prior_valid=np.concatenate([np.zeros(len(data.shops),bool),target_valid[:-len(data.shops)]])
    train_history=data.rows.target_date<=pd.Timestamp(test['train_end'])
    history_support=data.y[train_history].sum(axis=0)
    unsupported=np.setdiff1d(np.arange(len(data.names)),indices)
    novel=unsupported[history_support[unsupported]==0]
    seen_unsupported=unsupported[history_support[unsupported]>0]
    coverage={'expected_shop_day_forecasts':int(all_test.sum()),'scored':len(rows),'invalid_or_missing_target':int((all_test&~target_valid).sum()),'valid_target_invalid_previous_day':int((all_test&target_valid&~prior_valid).sum()),'supported_flavours':len(indices),'all_target_appearances':int(total.sum()),'supported_appearances':int(y.sum()),'unseen_flavour_appearances':int(data.y[rows][:,novel].sum()),'seen_but_below_support_appearances':int(data.y[rows][:,seen_unsupported].sum())}
    assert coverage['scored']+coverage['invalid_or_missing_target']+coverage['valid_target_invalid_previous_day']==coverage['expected_shop_day_forecasts']
    assert coverage['supported_appearances']+coverage['unseen_flavour_appearances']+coverage['seen_but_below_support_appearances']==coverage['all_target_appearances']
    per_label=[]
    for j,name in enumerate(bundle['labels']):
        for model,p in predictions.items():
            per_label.append({'flavour':name,'model':model,'train_support':int(bundle['train_support'][j]),'test_positives':int(y[:,j].sum()),'ap':float(average_precision_score(y[:,j],p[:,j])) if y[:,j].sum() else None,'brier':float(np.mean((y[:,j]-p[:,j])**2))})
    pd.DataFrame(per_label).to_csv(out/'per-flavour.csv',index=False)
    groups={}
    for group,mask in [('support_5_to_10',bundle['train_support']<=10),('support_above_10',bundle['train_support']>10)]:
        if mask.any():groups[group]={m:metrics(y[:,mask],p[:,mask],k=min(10,int(mask.sum()))) for m,p in predictions.items()}
    report={'period':test,'decision':decision,'metrics':results,'by_shop':per_shop,'coverage':coverage,'paired_bootstrap':intervals,'support_groups':groups,'bootstrap_method':f"Moving blocks of {config['bootstrap_block_days']} calendar days; both shops kept together; {config['bootstrap_replicates']} replicates; percentile 95% intervals; seed {config['seed']}. Conditional on this fitted model and test period; does not measure training or model-selection uncertainty."}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report(report,bundle,out)
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    manifest['outputs']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return decision


def write_report(report,bundle,out):
    m=report['metrics'];d=report['decision'];period=report['period']
    lines=['# Final held-out evaluation','',f"Training ends **{period['train_end']}**. Test: **{period['start']}–{period['end']}**. Model settings, transforms and vocabulary were frozen before the test. Daily features incorporate only earlier observations, including prior days within the test.",'',f"Recommendation: **{d['recommended']}**. {d['uncertainty']}.",'','| Predictor | Micro AP ↑ | Macro AP ↑ | Brier ↓ | Precision | Recall | k | All-flavour recall@k |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,v in m.items():lines.append(f"| {model} | {v['micro_ap']:.4f} | {v['macro_ap']:.4f} | {v['brier']:.5f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['k']} | {v['recall_at_k_all_flavours']:.3f} |")
    lines+=['','AP is a ranking metric, not classification accuracy. Thresholds and k are inherited from validation. Model retention uses the prespecified micro-AP/Brier gate; menu-recall results must be assessed separately.','', '## Paired uncertainty','',report['bootstrap_method'],'','Positive differences favor the selected candidate; Brier differences are baseline error minus candidate error.','', '| Baseline | AP gain, 95% interval | Brier reduction, 95% interval |','|---|---|---|']
    for b,v in report['paired_bootstrap'].items():
        ap=m[bundle['candidate']]['micro_ap']-m[b]['micro_ap'];br=m[b]['brier']-m[bundle['candidate']]['brier'];aci=v['ap_gain']['ci95'];bci=v['brier_reduction']['ci95']
        lines.append(f'| {b} | {ap:.4f} [{aci[0]:.4f}, {aci[1]:.4f}] | {br:.5f} [{bci[0]:.5f}, {bci[1]:.5f}] |')
    lines+=['','## Shop results','','| Shop | Predictor | Micro AP | Brier | All-flavour recall@k |','|---|---|---:|---:|---:|']
    for model,shops in report['by_shop'].items():
        for shop,v in shops.items():lines.append(f"| {shop} | {model} | {v['micro_ap']:.4f} | {v['brier']:.5f} | {v['recall_at_k_all_flavours']:.3f} |")
    lines+=['','## Coverage','','```json',json.dumps(report['coverage'],indent=2),'```','','Unseen and below-support flavours count as misses in all-flavour menu recall. Per-flavour metrics and training-support groups are saved locally. Macro AP excludes flavours with no test positives; counts are recorded in metrics.json. Missing and quarantined snapshots are not negatives.','', '## Calibration','','![Held-out reliability curves](final-calibration.png)','','## Reproducibility and limitations','','Pinned source: `'+bundle['source_commit']+'`. The local model bundle is `artifacts/final/model.joblib`; its training cutoff remains '+bundle['train_end']+'. Predictions, truth, selection hash, source/config/code/data hashes and checksums are saved under `artifacts/final/`. Rerunning final evaluation returns verified saved results, without retuning or rescoring.','', 'This is an independent final time period, conditional on the development protocol. Format inspection included test-period files during parser development; no model settings were tuned on test scores. Historical collection availability is assumed rather than timestamp-verified. Conservative name matching fragments some flavours. The closed vocabulary cannot identify new flavours. The saved model is local research software, not a guarantee of menu availability.']
    Path('reports/final-evaluation.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot([0,1],[0,1],color='gray',linestyle='--',label='Perfect calibration')
    for model,v in m.items():
        bins=[b for b in v['calibration'] if b['count']]
        ax.plot([b['mean_probability'] for b in bins],[b['observed_rate'] for b in bins],marker='o',label=model)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed appearance rate',title='Held-out probability calibration',xlim=(0,1),ylim=(0,1))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig('reports/final-calibration.png',dpi=160);plt.close(fig)
