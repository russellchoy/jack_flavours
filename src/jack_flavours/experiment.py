"""Explicit validation commands; final test evaluation is a separate stage."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import Dataset
from .evaluation import metrics,pooled,choose_operating_point


def fingerprint():
    files=sorted(Path('src').rglob('*.py'))+[Path('uv.lock'),Path('configs/experiment.json'),Path('configs/splits.json')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def frequency(data,train,indices,rows):
    train_y=data.y[train][:,indices]
    overall=train_y.mean(axis=0)
    shops=data.rows.shop.to_numpy()
    out=np.empty((len(rows),len(indices)),float)
    for shop in data.shops:
        sm=train & (shops==shop)
        rate=data.y[sm][:,indices].mean(axis=0) if sm.any() else overall
        out[shops[rows]==shop]=rate
    return out


def fold_record(data,fold,min_support):
    train=data.mask(end=fold['train_end']); mask=data.mask(fold['start'],fold['end']); rows=np.flatnonzero(mask)
    indices=data.vocabulary(train,min_support)
    if not train.any() or not len(rows) or not len(indices):raise ValueError('Empty fold')
    prev=rows-len(data.shops)
    all_y=data.y[rows]
    return train,{'name':fold['name'],'rows':rows,'indices':indices,'labels':np.array(data.names)[indices], 'dates':data.rows.target_date.to_numpy()[rows].astype('datetime64[D]').astype(str),'shops':data.rows.shop.to_numpy()[rows].astype(str),'y':all_y[:,indices],'total':all_y.sum(axis=1),'frequency':frequency(data,train,indices,rows),'persistence':data.y[prev][:,indices].astype(float)}


def summaries(records,models,config):
    output={}
    for model in models:
        point=choose_operating_point(records,model,config['top_k'],config['thresholds'])
        folds=[]
        for r in records:
            by_shop={}
            for shop in sorted(set(r['shops'])):
                mask=r['shops']==shop
                by_shop[shop]=metrics(r['y'][mask],r[model][mask],r['total'][mask],**point)
            folds.append({'name':r['name'],'metrics':metrics(r['y'],r[model],r['total'],**point),'by_shop':by_shop})
        output[model]={'pooled':pooled(records,model),'operating_point':point,'folds':folds}
    return output


def save_records(records,out):
    out.mkdir(parents=True,exist_ok=True)
    for r in records:
        np.savez_compressed(out/(r['name']+'.npz'),**{k:v for k,v in r.items() if k!='name'})


def baselines(config):
    data=Dataset.load(Path('data/processed'),config['shops'])
    split=json.loads(Path('configs/splits.json').read_text())
    records=[fold_record(data,f,config['min_support'])[1] for f in split['folds']]
    result=summaries(records,['frequency','persistence'],config)
    out=Path('artifacts/baselines');save_records(records,out)
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={'stage':'validation_baselines','source_commit':config['source_commit'],'files':fingerprint(),'test_scored':False,'eligible_validation_forecasts':sum(len(r['rows']) for r in records),'availability_assumption':config['cutoff']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Validation baselines','','Final test remains untouched. All results are on the three rolling validation windows.','', '| Predictor | Micro AP (higher is better) | Brier (lower is better) | Chosen k | Threshold |','|---|---:|---:|---:|---:|']
    for model,v in result.items():lines.append(f"| {model} | {v['pooled']['micro_ap']:.4f} | {v['pooled']['brier']:.4f} | {v['operating_point']['k']} | {v['operating_point']['threshold']} |")
    lines+=['','Both predictors use the same forecast rows and training-only vocabulary in each fold. Missing target or previous-day snapshots are not scored. Threshold and k are selected on validation only. Precision/recall at k counts out-of-vocabulary flavours as missed. Full fold/shop metrics and calibration bins are in local artifacts/baselines/metrics.json.','', '| Fold | Forecasts | Eligible labels | Actual flavour appearances | In-vocabulary appearances |','|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"| {r['name']} | {len(r['rows'])} | {len(r['indices'])} | {r['total'].sum()} | {r['y'].sum()} |")
    lines+=['','Macro AP omits labels without validation positives and reports the eligible count. Empty target recall is defined as zero (valid empty menus are excluded by the parser). Stable lexical ordering breaks probability ties. No model claims are made from these baseline results.']
    Path('reports/baselines.md').write_text('\n'.join(lines)+'\n')
    return {m:v['pooled'] for m,v in result.items()}


def validate_models(config):
    from datetime import datetime, timezone
    import warnings
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import average_precision_score
    from threadpoolctl import threadpool_limits
    from .models import fit_binary,choose_model
    deadline=datetime.fromisoformat(config['session']['deadline_utc'])
    def check_deadline():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Session deadline reached; do not extend automatically')
    check_deadline()
    out=Path('artifacts/validation');out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Validation selection already frozen; do not silently rerun or overwrite')
    data=Dataset.load(Path('data/processed'),config['shops'])
    split=json.loads(Path('configs/splits.json').read_text())
    records=[]; feature_cache={}; per_label=[]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for fold in split['folds']:
            check_deadline()
            train,r=fold_record(data,fold,config['min_support'])
            rows=r['rows'];train_rows=np.flatnonzero(train)
            models=[f'logit_C{C:g}' for C in config['logistic_C']]+['boosting']
            for model in models:r[model]=np.empty(r['y'].shape,dtype=float)
            for j,index in enumerate(r['indices']):
                check_deadline()
                if index not in feature_cache:feature_cache[index]=data.features(index)
                x=feature_cache[index]; y=data.y[train_rows,index]
                for C in config['logistic_C']:
                    fitted=fit_binary(x[train_rows],y,'logit',config,C)
                    r[f'logit_C{C:g}'][:,j]=fitted.predict_proba(x[rows])[:,1]
                fitted=fit_binary(x[train_rows],y,'boosting',config)
                r['boosting'][:,j]=fitted.predict_proba(x[rows])[:,1]
                if (j+1)%100==0:print(f"{fold['name']}: fitted {j+1}/{len(r['indices'])} flavour models",flush=True)
            records.append(r)
            save_records([r],out)
            print(f"{fold['name']} saved; test untouched",flush=True)
    ranking={f'logit_C{C:g}':pooled(records,f'logit_C{C:g}') for C in config['logistic_C']}
    best_C=max(config['logistic_C'],key=lambda c:(ranking[f'logit_C{c:g}']['micro_ap'],-ranking[f'logit_C{c:g}']['brier'],-c))
    for r in records:r['logit']=r[f'logit_C{best_C:g}']
    summary=summaries(records,['frequency','persistence','logit','boosting'],config)
    decision=choose_model(summary)
    decision.update({'logistic_C':best_C,'operating_point':summary[decision['selected']]['operating_point'],'source_commit':config['source_commit'],'files':fingerprint(),'test_period':split['test'],'validation_metrics':{m:v['pooled'] for m,v in summary.items()},'min_support':config['min_support'],'boosting_parameters':config['boosting']})
    # Per-flavour analysis stays local; never read test labels to choose parameters.
    for r in records:
        train=data.mask(end=next(f['train_end'] for f in split['folds'] if f['name']==r['name']))
        for j,index in enumerate(r['indices']):
            for model in ['frequency','persistence','logit','boosting']:
                truth=r['y'][:,j];p=r[model][:,j]
                per_label.append({'fold':r['name'],'flavour':data.names[index],'model':model,'train_positives':int(data.y[train,index].sum()),'validation_positives':int(truth.sum()),'ap':float(average_precision_score(truth,p)) if truth.sum() else None,'brier':float(np.mean((truth-p)**2))})
    pd.DataFrame(per_label).to_csv(out/'per-flavour.csv',index=False)
    save_records(records,out)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'regularisation-search.json').write_text(json.dumps(ranking,indent=2)+'\n')
    (out/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'stage':'model_validation','source_commit':config['source_commit'],'files':fingerprint(),'test_scored':False,'started_session':config['session'],'completed_utc':datetime.now(timezone.utc).isoformat(),'availability_assumption':config['cutoff']},indent=2)+'\n')
    write_validation_report(summary,decision,ranking,records)
    return {k:v for k,v in decision.items() if k!='files'}


def write_validation_report(summary,decision,ranking,records):
    lines=['# Model validation results','','These results use rolling validation only. The final 90-day test has not been evaluated; they are model-selection evidence, not an unbiased final performance estimate.','',f"Selected for the later final evaluation: **{decision['selected']}**. Best logistic C: **{decision['logistic_C']}** (smaller C means stronger L2 regularisation).",'', '| Predictor | Pooled micro AP ↑ | Brier ↓ | k | Threshold |','|---|---:|---:|---:|---:|']
    for m,v in summary.items():lines.append(f"| {m} | {v['pooled']['micro_ap']:.4f} | {v['pooled']['brier']:.4f} | {v['operating_point']['k']} | {v['operating_point']['threshold']} |")
    lines+=['','## Stability across validation periods','','| Period | Predictor | Micro AP | Brier | Menu recall@k including unseen flavours |','|---|---|---:|---:|---:|']
    for m,v in summary.items():
        for f in v['folds']:
            z=f['metrics'];lines.append(f"| {f['name']} | {m} | {z['micro_ap']:.4f} | {z['brier']:.4f} | {z['recall_at_k_all_flavours']:.4f} |")
    lines+=['','## Decision evidence','','```json',json.dumps({k:decision[k] for k in ['selected','baseline_fallback','logit_passes_validation_gate','boosting_passes_validation_gate','boosting_substantially_beats_logit','boosting_better_folds','boosting_better_both_shops','final_test_status']},indent=2),'```','','## Regularisation comparison','','| C | Validation micro AP | Brier |','|---|---:|---:|']
    for m,v in ranking.items():lines.append(f"| {m.removeprefix('logit_C')} | {v['micro_ap']:.4f} | {v['brier']:.4f} |")
    lines+=['','## Coverage and limitations','','| Period | Eligible forecasts | Supported labels | Coverage of actual flavour appearances |','|---|---:|---:|---:|']
    for r in records:lines.append(f"| {r['name']} | {len(r['rows'])} | {len(r['indices'])} | {r['y'].sum()/r['total'].sum():.1%} |")
    lines+=['','Flavour vocabulary requires at least five positives in eligible training snapshots. Unknown and low-support flavours are not forecast, but count as misses in all-flavour menu recall. Canonical names are intentionally conservative; spelling variants may fragment labels. Brier scores average many negative labels, so read them alongside AP and calibration. Historical collection availability is assumed, not verified. Quarantined dates and missing inputs reduce coverage.','', 'Model selection and operating points used these validation windows, so the reported advantage may be optimistic. Per-shop metrics, calibration bins and per-flavour results (including training support) are saved in ignored artifacts/validation. No semantic alias tuning or test-driven parameter changes were made.','', '## Remaining work','','The six-attempt session limit was reached after validation (one parser-test fixture repair consumed an attempt). Final refitting, held-out scoring, paired date-block uncertainty and a deployable model artifact remain pending a separately authorized continuation. The frozen selection and predictions are saved; do not retune them on the final test.']
    Path('reports/model-validation.md').write_text('\n'.join(lines)+'\n')
