"""One predeclared exploratory co-occurrence ablation; no model promotion."""
from pathlib import Path
import os,time,json,warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from jack_flavours.features import Dataset
from jack_flavours.peer_features import select_peers,peer_features
from jack_flavours.models import fit_binary
from jack_flavours.evaluation import metrics
from jack_flavours.final import digest,sample_calendar_blocks
ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);start=time.monotonic()
    def check():
        if time.monotonic()-start>=1800:raise TimeoutError('30-minute experiment limit reached')
    out=Path('artifacts/cooccurrence-experiment')
    if out.exists():raise ValueError('Experiment directory exists; inspect before any retry')
    out.mkdir()
    protected={str(p):digest(p) for directory in ['artifacts/final','artifacts/validation','artifacts/latest','data/processed'] for p in Path(directory).iterdir() if p.is_file()}
    config=json.loads(Path('configs/experiment.json').read_text());splits=json.loads(Path('configs/splits.json').read_text())
    data=Dataset.load(Path('data/cleaned-descriptive-v3'),config['shops'])
    manifest={'plan_sha256':digest('reports/cooccurrence-implementation-plan.md'),'parameters':{'C':.01,'min_support':5,'peers_per_shop':5,'peer_min_support':20,'peer_min_joint':10,'peer_lift_min_exclusive':1,'bootstrap_replicates':200,'block_days':7},'input_sha256':{str(p):digest(p) for p in Path('data/cleaned-descriptive-v3').glob('*.parquet')},'code_sha256':{p:digest(p) for p in ['src/jack_flavours/peer_features.py','src/jack_flavours/features.py','src/jack_flavours/models.py','scripts/cooccurrence_experiment.py','uv.lock']},'status':'running','limitations':'Exploratory reused historical dates and full-history-informed frozen v3 alias rules; no independent test or accuracy claim.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]; results={}
    for fold in splits['folds']+[dict(name='reused_holdout',**splits['test'])]:
        check();name=fold['name'];train=data.mask(end=fold['train_end']);test=data.mask(fold['start'],fold['end']);indices=data.vocabulary(train,5)
        with threadpool_limits(limits=1):peers,peer_rows=select_peers(data,fold['train_end'],indices)
        pd.DataFrame(peer_rows).to_csv(out/f'{name}-peers.csv',index=False)
        y=data.y[test][:,indices];ps={m:np.zeros(y.shape) for m in ['baseline','peers']}
        with threadpool_limits(limits=1),warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            for j,index in enumerate(indices):
                check();x=data.features(index);extra=peer_features(data,peers[int(index)])
                for model,features in [('baseline',x),('peers',np.column_stack([x,extra]))]:
                    fitted=fit_binary(features[train],data.y[train,index],'logit',config,C=.01)
                    ps[model][:,j]=fitted.predict_proba(features[test])[:,1]
        dates=data.rows.loc[test,'target_date'].to_numpy();shops=data.rows.loc[test,'shop'].to_numpy();total=data.y[test].sum(axis=1)
        record={'y':y,**ps,'dates':dates,'shops':shops,'total':total,'labels':np.array(data.names)[indices]};records.append(record)
        np.savez_compressed(out/f'{name}-predictions.npz',**record)
        results[name]={'window':fold,'selected_peer_slots':len(peer_rows),'targets_with_peers':sum(any(v.values()) for v in peers.values()),'targets':len(indices),'models':{m:metrics(y,p,total,k=10,threshold=.2) for m,p in ps.items()},'by_shop':{shop:{m:metrics(y[shops==shop],p[shops==shop],total[shops==shop],k=10,threshold=.2) for m,p in ps.items()} for shop in data.shops}}
        (out/'window-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,{m:round(results[name]['models'][m]['micro_ap'],5) for m in ps},flush=True)
    def pooled(rs,draw=None):
        yy=[];pp={m:[] for m in ['baseline','peers']};rec={m:[] for m in pp}
        all_dates=np.concatenate([r['dates'] for r in rs]);offset=0
        for r in rs:
            take=np.arange(len(r['y'])) if draw is None else draw[(draw>=offset)&(draw<offset+len(r['y']))]-offset
            offset+=len(r['y'])
            if not len(take):continue
            truth=r['y'][take]; yy.append(truth.ravel())
            for m in pp:
                prediction=r[m][take];pp[m].append(prediction.ravel())
                top=np.argsort(-prediction,axis=1,kind='stable')[:,:10]
                rec[m].extend(np.take_along_axis(truth,top,axis=1).sum(axis=1)/r['total'][take])
        y=np.concatenate(yy)
        return {m:{'micro_ap':float(average_precision_score(y,np.concatenate(pp[m]))),'brier':float(np.mean((y-np.concatenate(pp[m]))**2)),'recall_at_10':float(np.mean(rec[m]))} for m in pp}
    aggregate={};intervals={}
    for label,rs in [('validation',records[:3]),('reused_holdout',records[3:])]:
        aggregate[label]=pooled(rs);dates=np.concatenate([r['dates'] for r in rs]);rng=np.random.default_rng(42);draws=[]
        for b in range(200):
            check();ix=sample_calendar_blocks(dates,str(pd.Timestamp(dates.min()).date()),str(pd.Timestamp(dates.max()).date()),7,rng)
            if not len(ix):continue
            m=pooled(rs,ix);draws.append([m['peers']['micro_ap']-m['baseline']['micro_ap'],m['baseline']['brier']-m['peers']['brier'],m['peers']['recall_at_10']-m['baseline']['recall_at_10']])
        intervals[label]={metric:np.quantile(np.array(draws)[:,i],[.025,.975]).tolist() for i,metric in enumerate(['ap_gain','brier_reduction','recall_at_10_gain'])}
    v=aggregate['validation'];h=aggregate['reused_holdout'];wins=sum(results[f['name']]['models']['peers']['micro_ap']>results[f['name']]['models']['baseline']['micro_ap'] for f in splits['folds'])
    passes=v['peers']['micro_ap']-v['baseline']['micro_ap']>=.005 and v['peers']['brier']<=v['baseline']['brier'] and wins>=2 and h['peers']['micro_ap']-h['baseline']['micro_ap']>=.005 and h['peers']['brier']<=h['baseline']['brier'] and all(g['peers']['micro_ap']-g['baseline']['micro_ap']>=-.01 for g in results['reused_holdout']['by_shop'].values())
    for p,sha in protected.items():assert digest(p)==sha,p
    for p,sha in manifest['input_sha256'].items():assert digest(p)==sha,p
    summary={'aggregate':aggregate,'paired_block_ci95':intervals,'validation_ap_wins':wins,'promising_screen_passed':bool(passes),'recommendation':'Prospective confirmation warranted; no promotion' if passes else 'Keep baseline; this fixed co-occurrence encoding did not pass the screen','elapsed_seconds':time.monotonic()-start,'preserved_artifacts_verified':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');manifest.update(status='complete',protected_sha256=protected);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
