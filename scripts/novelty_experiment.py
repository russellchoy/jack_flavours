"""Fixed exploratory ablation; original models/artifacts remain untouched."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from jack_flavours.features import Dataset
from jack_flavours.models import fit_binary,ConstantProbability
from jack_flavours.evaluation import metrics
from jack_flavours.final import sample_calendar_blocks


def novelty_data(data):
    days,shops,flavours=data.presence.shape
    seen=np.zeros(flavours,bool);local=np.zeros((shops,flavours),bool)
    counts=np.zeros((days,shops));local_counts=np.zeros_like(counts)
    for t in range(days):
        current=data.presence[t].astype(bool)&data.observed[t,:,None]
        counts[t]=(current&~seen).sum(axis=1)
        local_counts[t]=(current&~local).sum(axis=1)
        seen|=current.any(axis=0);local|=current
    blocks=[]
    weekday=data.dates.dayofweek.to_numpy();month=data.dates.month.to_numpy()
    for s in range(shops):
        values=counts[:,s].copy();values[~data.observed[:,s]]=np.nan
        event=pd.Series(np.where(np.isnan(values),np.nan,values>0))
        lag=event.shift(1);count=pd.Series(values).shift(1)
        age=np.full(days,999.);last=None
        for t in range(days):
            if last is not None:age[t]=t-last
            if not np.isnan(values[t]) and values[t]>0:last=t
        turn=np.full(days,np.nan)
        for t in range(1,days):
            if data.observed[t,s] and data.observed[t-1,s]:
                turn[t]=np.count_nonzero(data.presence[t,s]!=data.presence[t-1,s])
        lagturn=pd.Series(turn).shift(1)
        size=pd.Series(np.where(data.observed[:,s],data.presence[:,s].sum(axis=1),np.nan)).shift(1)
        cols=[np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),np.sin(2*np.pi*(month-1)/12),np.cos(2*np.pi*(month-1)/12),np.full(days,s),size.fillna(0),lag.fillna(0),count.fillna(0),age,lagturn.fillna(0),lagturn.isna()]
        for window in [7,30,90]:
            cols.extend([lag.rolling(window,min_periods=1).mean().fillna(.5),count.rolling(window,min_periods=1).mean().fillna(0),lag.rolling(window,min_periods=1).count().fillna(0)])
        blocks.append(np.column_stack(cols))
    x=np.stack(blocks,axis=1).reshape(-1,len(blocks[0][0])).astype(float)
    return x,(counts>0).astype(int).ravel(),counts.ravel(),local_counts.ravel()


def status_features(data,index):
    days,shops,_=data.presence.shape
    blocks=[]
    for s in range(shops):
        seen=0;returns=0;first=None;rows=[]
        for t in range(days):
            lag_present=t>0 and data.observed[t-1,s] and bool(data.presence[t-1,s,index])
            rows.append([np.log1p(seen),np.log1p(returns),999 if first is None else t-first,int(lag_present and returns==0),int(returns>0)])
            if data.observed[t,s] and data.presence[t,s,index]:
                if seen and t>0 and data.observed[t-1,s] and not data.presence[t-1,s,index]:returns+=1
                if first is None:first=t
                seen+=1
        blocks.append(rows)
    return np.stack(blocks,axis=1).reshape(-1,5).astype(float)


def novelty_eligible(data,burn_in):
    return data.eligible & np.repeat(np.arange(len(data.dates))>=burn_in,len(data.shops))


def train_novelty(x,y,mask,config):
    if mask.sum()<config['min_novelty_training_rows']:
        return ConstantProbability(float((y[mask].sum()+1)/(mask.sum()+2)))
    return fit_binary(x[mask],y[mask],'logit',config,config['logistic_C'])


def prequential(x,y,data,config):
    """Training meta-features: model cutoff strictly precedes each 30-day block."""
    eligible=novelty_eligible(data,config['burn_in_days'])
    p=np.empty(len(y));cutoff=np.empty(len(y),dtype=int)
    nshops=len(data.shops)
    for start in range(0,len(data.dates),config['inner_block_days']):
        stop=min(start+config['inner_block_days'],len(data.dates))
        train=eligible & (np.arange(len(y))<start*nshops)
        model=train_novelty(x,y,train,config)
        sl=slice(start*nshops,stop*nshops)
        p[sl]=model.predict_proba(x[sl])[:,1]
        cutoff[sl]=start-1
    return p,cutoff


def simple(y,p):
    y=np.asarray(y).ravel();p=np.asarray(p).ravel()
    return {'ap':float(average_precision_score(y,p)) if y.sum() else None,'brier':float(np.mean((y-p)**2)),'pairs':len(y),'positive_rate':float(y.mean())}


def paired_interval(y,a,b,dates,start,end,config,check):
    rng=np.random.default_rng(config['seed']);draws=[]
    for _ in range(config['bootstrap_replicates']):
        check();idx=sample_calendar_blocks(dates,start,end,7,rng)
        if not len(idx) or not y[idx].sum():continue
        ys=y[idx].ravel();ap=a[idx].ravel();bp=b[idx].ravel()
        draws.append([average_precision_score(ys,ap)-average_precision_score(ys,bp),np.mean((ys-bp)**2)-np.mean((ys-ap)**2)])
    if len(draws)<config['bootstrap_replicates']//2:raise ValueError('Insufficient informative bootstrap draws')
    return {'ap_gain_ci95':np.quantile(np.array(draws)[:,0],[.025,.975]).tolist(),'brier_reduction_ci95':np.quantile(np.array(draws)[:,1],[.025,.975]).tolist(),'replicates':len(draws)}


def run():
    config=json.loads(Path('configs/novelty-experiment.json').read_text())
    def check():
        if datetime.now(timezone.utc)>=datetime.fromisoformat(config['deadline_utc']):raise TimeoutError('Novelty session deadline reached')
    check()
    out=Path('artifacts/novelty-experiment')
    if out.exists():raise ValueError('Experiment output exists; inspect before rerunning, do not overwrite')
    exp=json.loads(Path('configs/experiment.json').read_text());splits=json.loads(Path('configs/splits.json').read_text())
    data=Dataset.load(Path('data/processed'),exp['shops'])
    saved_paths=[p for directory in ['artifacts/validation','artifacts/final'] for p in Path(directory).iterdir() if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in saved_paths}
    out.mkdir(parents=True)
    xnov,ynov,ncounts,localcounts=novelty_data(data)
    eligible_nov=novelty_eligible(data,config['burn_in_days'])
    records=[];novrecords=[];cache={}
    folds=splits['folds']+[{'name':'exploratory_final',**splits['test']}]
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        preq,cutoffs=prequential(xnov,ynov,data,config)
        assert (cutoffs<np.repeat(np.arange(len(data.dates)),len(data.shops))).all()
        for fold in folds:
            check();train=data.mask(end=fold['train_end']);test=data.mask(fold['start'],fold['end']);rows=np.flatnonzero(test)
            indices=data.vocabulary(train,exp['min_support']);truth=data.y[rows][:,indices];total=data.y[rows].sum(axis=1)
            saved=np.load('artifacts/final/predictions.npz' if fold['name']=='exploratory_final' else f"artifacts/validation/{fold['name']}.npz",allow_pickle=False)
            np.testing.assert_array_equal(truth,saved['y']);np.testing.assert_array_equal(np.array(data.names)[indices],saved['labels'])
            dates=data.rows.iloc[rows].target_date.to_numpy().astype('datetime64[D]').astype(str)
            np.testing.assert_array_equal(dates,saved['dates'])
            record={'name':fold['name'],'y':truth,'total':total,'dates':dates,'shops':data.rows.iloc[rows].shop.to_numpy().astype(str),'current':saved['logit'].copy(),'status_only':np.empty_like(saved['logit']),'integrated':np.empty_like(saved['logit'])}
            ntrain=train & eligible_nov; ntest=test & eligible_nov;nrows=np.flatnonzero(ntest)
            nmodel=train_novelty(xnov,ynov,ntrain,config)
            npred=nmodel.predict_proba(xnov[rows])[:,1]
            freq=np.empty(len(nrows))
            allshops=data.rows.shop.to_numpy()
            for shop in data.shops:
                mask=ntrain & (allshops==shop)
                freq[allshops[nrows]==shop]=ynov[mask].mean() if mask.any() else ynov[ntrain].mean()
            # The lagged 7/30/90-day feature groups start at columns 11, 14 and 17.
            recent=xnov[nrows,14]
            novrecord={'name':fold['name'],'y':ynov[nrows,None],'dates':data.rows.iloc[nrows].target_date.to_numpy().astype('datetime64[D]').astype(str),'shops':allshops[nrows].astype(str),'frequency':freq[:,None],'recent30':recent[:,None],'novelty_logit':nmodel.predict_proba(xnov[nrows])[:,1,None],'new_to_shop_positive_rate':float((localcounts[nrows]>0).mean())}
            novrecords.append(novrecord)
            # All inner training predictions come from models fitted BEFORE their target block.
            for j,index in enumerate(indices):
                check()
                if index not in cache:cache[index]=(data.features(index),status_features(data,index))
                base,status=cache[index];statusx=np.column_stack([base,status]);y=data.y[:,index]
                smodel=fit_binary(statusx[train],y[train],'logit',config,config['logistic_C'])
                record['status_only'][:,j]=smodel.predict_proba(statusx[rows])[:,1]
                trainx=np.column_stack([statusx[train],preq[train]])
                evalx=np.column_stack([statusx[rows],npred])
                imodel=fit_binary(trainx,y[train],'logit',config,config['logistic_C'])
                record['integrated'][:,j]=imodel.predict_proba(evalx)[:,1]
            records.append(record)
            np.savez_compressed(out/(fold['name']+'-known.npz'),**{k:v for k,v in record.items() if k!='name'})
            np.savez_compressed(out/(fold['name']+'-novelty.npz'),**{k:v for k,v in novrecord.items() if k!='name'})
            print(f"Completed {fold['name']}: {len(rows)} forecasts, {len(indices)} known flavours",flush=True)
    result={'known':{},'novelty':{},'folds':{},'intervals':{},'settings':config}
    for name,subset in [('validation',slice(0,3)),('exploratory_final',slice(3,4))]:
        result['known'][name]={}
        for model in ['current','status_only','integrated']:
            selected=records[subset];y=np.concatenate([r['y'].ravel() for r in selected]);p=np.concatenate([r[model].ravel() for r in selected])
            score=simple(y,p)
            score['menu_recall_at_10']=sum(metrics(r['y'],r[model],r['total'],**config['known_flavour_point'])['recall_at_k_all_flavours']*len(r['y']) for r in selected)/sum(len(r['y']) for r in selected)
            result['known'][name][model]=score
        result['novelty'][name]={m:simple(np.concatenate([r['y'] for r in novrecords[subset]]),np.concatenate([r[m] for r in novrecords[subset]])) for m in ['frequency','recent30','novelty_logit']}
    for r,n in zip(records,novrecords):
        result['folds'][r['name']]={'known':{m:simple(r['y'],r[m]) for m in ['current','status_only','integrated']},'novelty':{m:simple(n['y'],n[m]) for m in ['frequency','recent30','novelty_logit']},'by_shop':{shop:{'known':{m:simple(r['y'][r['shops']==shop],r[m][r['shops']==shop]) for m in ['current','status_only','integrated']},'novelty':{m:simple(n['y'][n['shops']==shop],n[m][n['shops']==shop]) for m in ['frequency','recent30','novelty_logit']}} for shop in data.shops}}
    last=records[-1];nlast=novrecords[-1]
    for model in ['status_only','integrated']:
        result['intervals'][model+'_vs_current']=paired_interval(last['y'],last[model],last['current'],last['dates'],splits['test']['start'],splits['test']['end'],config,check)
    for base in ['frequency','recent30']:
        result['intervals']['novelty_vs_'+base]=paired_interval(nlast['y'],nlast['novelty_logit'],nlast[base],nlast['dates'],splits['test']['start'],splits['test']['end'],config,check)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({'original_artifact_hashes':hashes,'config_hash':hashlib.sha256(Path('configs/novelty-experiment.json').read_bytes()).hexdigest(),'script_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_commit':exp['source_commit'],'original_artifacts_unchanged':True,'final_period_independent':False},indent=2)+'\n')
    write_report(result)
    print(json.dumps({k:result[k] for k in ['known','novelty','intervals']},indent=2))


def write_report(result):
    lines=['# Does novelty separation improve prediction?','','Exploratory fixed comparison, 17 October 2026. These historical periods were already inspected; this is not a new independent test. Production model and original results are unchanged.','','## Known-flavour comparison','','Same forecast rows, training vocabulary, C=0.01, threshold=0.2 and k=10. A standalone novelty forecast leaves existing known-flavour probabilities unchanged by definition. Status-only adds past appearance count, confirmed return count, first-observed age, first-run indicator and returning indicator. Integrated adds an out-of-time novelty probability.','','| Period | Model | Micro AP ↑ | Brier ↓ | All-flavour recall@10 ↑ |','|---|---|---:|---:|---:|']
    for period,models in result['known'].items():
        for model,m in models.items():lines.append(f"| {period} | {model} | {m['ap']:.5f} | {m['brier']:.6f} | {m['menu_recall_at_10']:.3%} |")
    lines+=['','## New-flavour event prediction','','Target: at least one flavour at this shop tomorrow never previously recorded at either shop. Simultaneous first appearances count at both shops. This predicts an event, not names. First 90 days are history-building, excluded from fitting novelty targets. “New” means first recorded, not necessarily a genuinely new recipe. Unknown snapshot days can hide introductions; conservative naming can create false novelty.','','| Period | Predictor | AP ↑ | Brier ↓ | Event rate |','|---|---|---:|---:|---:|']
    for period,models in result['novelty'].items():
        for model,m in models.items():lines.append(f"| {period} | {model} | {m['ap']:.4f} | {m['brier']:.4f} | {m['positive_rate']:.1%} |")
    lines+=['','## Paired uncertainty on reused final period','','200 moving-block bootstrap replicates, seven calendar days, both shops sampled together, seed 42. Positive differences favour the added model. Intervals do not remove the exploratory nature of this comparison.','','| Comparison | AP gain 95% interval | Brier reduction 95% interval |','|---|---|---|']
    for name,m in result['intervals'].items():lines.append(f"| {name} | [{m['ap_gain_ci95'][0]:.5f}, {m['ap_gain_ci95'][1]:.5f}] | [{m['brier_reduction_ci95'][0]:.6f}, {m['brier_reduction_ci95'][1]:.6f}] |")
    lines+=['','## Temporal safeguards and scope','','Outer windows match the original three expanding validation windows and reused 90-day final period. Novelty models use only outer-training targets and past features. Training meta-features are generated in expanding 30-day blocks fitted strictly before each block; before enough novelty-training examples exist, a past-only smoothed frequency fallback is used. Each evaluation window uses a novelty model fitted at its outer training cutoff, with fixed weights. Neither actual target novelty nor future first-appearance dates enter features. Missing snapshots do not create artificial returns.','', 'All model settings were fixed before this run; no hyperparameter search or production replacement. No novel names were removed from the denominator of all-flavour menu recall. Known-flavour label support stays identical to the original experiment. Results and per-shop/fold metrics are in artifacts/novelty-experiment/results.json; paired predictions and unchanged-original hashes are also saved there.','', 'Reproduce with `.venv/bin/python scripts/novelty_experiment.py` in a fresh experiment output directory and an explicitly authorized time window. Existing output is protected against overwriting. Independent confirmation would require newly collected future data and a prospective protocol.']
    Path('reports/novelty-experiment.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':run()
