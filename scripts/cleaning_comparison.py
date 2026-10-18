"""Compare approved cleaning stages with fixed latest-history model settings."""
from pathlib import Path
from copy import deepcopy
import json, os, time
import joblib
import numpy as np
import pandas as pd
from jack_flavours.cleaning import clean_appearances, canonical_title, normalize_title
from jack_flavours.features import Dataset
from jack_flavours.final import fit_bundle, digest
from jack_flavours.forecast import build_forecast
from flavour_descriptives import summarise, menu_similarity, menu_autocorrelogram

ROOT=Path(__file__).resolve().parents[1]

def main():
    os.chdir(ROOT)
    out=Path('artifacts/cleaning-comparison-v3')
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def check():
        if time.monotonic()-start>1800:raise TimeoutError('30-minute comparison limit reached; no automatic extension')
    preserved={str(p):digest(p) for d in ['artifacts/final','artifacts/validation','artifacts/latest','data/processed'] for p in Path(d).iterdir() if p.is_file()}
    config=json.loads(Path('configs/experiment.json').read_text())
    old=joblib.load('artifacts/latest/model.joblib')
    snapshots=pd.read_parquet('data/processed/snapshots.parquet');raw=pd.read_parquet('data/processed/appearances.parquet')
    frames={}; summaries={}; stage_data={}; stage_meta={}
    for stage,table in [('original',raw),('conjunctions',clean_appearances(raw,confirmed_aliases=False)),('confirmed',clean_appearances(raw))]:
        data=Dataset.from_tables(snapshots,table,config['shops']); stage_data[stage]=data
        stats,events=summarise(data)
        stats.to_csv(out/f'{stage}-flavour-statistics.csv',index=False)
        similarity=menu_similarity(data);similarity.to_csv(out/f'{stage}-daily-similarity.csv',index=False)
        lag=menu_autocorrelogram(data);lag.to_csv(out/f'{stage}-autocorrelogram.csv',index=False)
        complete=events[(events.kind=='stay')&events.complete]
        strict=events[(events.kind=='absence')&events.complete]
        recorded=events[events.kind=='recorded_return']
        summaries[stage]={'distinct_flavours':len(data.names),'appearances':int(data.presence.sum()),'valid_menus':int(data.observed.sum()),'completed_stay_mean':float(complete.days.mean()),'completed_stay_median':float(complete.days.median()),'completed_stays':len(complete),'strict_gap_mean':float(strict.days.mean()),'recorded_return_mean':float(recorded.days.mean()),'cross_shop_similarity_mean':float(similarity.similarity_pct.mean()),'cross_shop_similarity_median':float(similarity.similarity_pct.median())}
        directory=out/stage;directory.mkdir(exist_ok=True)
        if stage=='original':bundle=old
        else:
            # Refuse silent overwrite/reuse: each comparison is a fresh, bounded fit.
            if (directory/'model.joblib').exists():raise ValueError(f'{directory} already fitted; inspect before rerunning')
            selection=deepcopy(old['selection']);selection['selected']=old['recommended'];selection['test_period']={'train_end':str(data.dates[-1].date())}
            bundle=fit_bundle(data,selection,config,check)
            bundle.update(recommended=old['recommended'],operating_point=old['operating_point'],training_role=f'cleaning_comparison_{stage}',normalization_version=stage,not_independently_evaluated=True)
            joblib.dump(bundle,directory/'model.joblib')
            table.to_parquet(directory/'appearances.parquet',index=False)
            snapshots.to_parquet(directory/'snapshots.parquet',index=False)
        forecast,meta=build_forecast(bundle,data);frames[stage]=forecast;stage_meta[stage]=meta
        forecast.to_csv(directory/'forecast.csv',index=False);forecast[forecast.top_k].to_csv(directory/'top10.csv',index=False)
        meta.update({'model_sha256':digest('artifacts/latest/model.joblib' if stage=='original' else directory/'model.joblib'),'normalization_version':stage,'not_independently_evaluated':True})
        (directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        summaries[stage]['supported_labels']=len(bundle['labels'])
    def compare(before,after,mapper):
        left=frames[before].copy();left['canonical_flavour']=left.flavour.map(mapper)
        rows=[]
        for r in frames[after].itertuples():
            match=left[(left.shop==r.shop)&(left.canonical_flavour==r.flavour)]
            rows.append({'shop':r.shop,'flavour':r.flavour,'old_labels':' | '.join(match.flavour),'old_probabilities':' | '.join(f'{p:.8f}' for p in match.probability),'old_label_count':len(match),'old_union_lower':float(match.probability.max()) if len(match) else np.nan,'old_union_upper':float(min(1,match.probability.sum())) if len(match) else np.nan,'new_probability':r.probability,'new_rank':r.rank,'old_best_rank':int(match['rank'].min()) if len(match) else np.nan,'new_top10':r.top_k,'old_top10':bool(match.top_k.any()),'change_pp_single_label':100*(r.probability-match.probability.iloc[0]) if len(match)==1 else np.nan})
        comp=pd.DataFrame(rows);comp.to_csv(out/f'{before}-to-{after}-forecast.csv',index=False)
        info={}
        for shop,g in comp.groupby('shop'):
            oldset=set(left.loc[(left.shop==shop)&left.top_k,'canonical_flavour']);newset=set(g.loc[g.new_top10,'flavour'])
            info[shop]={'retained_top10':len(oldset&newset),'old_unique_top10':len(oldset),'entrants':sorted(newset-oldset),'departures':sorted(oldset-newset)}
        return info
    comparisons={'original_to_confirmed':compare('original','confirmed',canonical_title),'conjunctions_to_confirmed':compare('conjunctions','confirmed',canonical_title)}
    for p,sha in preserved.items():assert digest(p)==sha,p
    manifest={'stages':summaries,'top10_comparisons':comparisons,'target_date':stage_meta['confirmed']['target_date'],'train_end':old['train_end'],'parameters':{'logistic_C':old['selection']['logistic_C'],'min_support':old['selection']['min_support'],'operating_point':old['operating_point']},'preserved_sha256':preserved,'implementation_sha256':{p:digest(p) for p in ['src/jack_flavours/cleaning.py','scripts/cleaning_comparison.py','src/jack_flavours/features.py','src/jack_flavours/models.py','uv.lock']},'elapsed_seconds':time.monotonic()-start,'interpretation':'Forecast changes only; no independent accuracy evaluation. Old alias probabilities define distinct historical labels: their union is bounded by max(p) and min(1,sum(p)), not an exact summed probability. Missing old labels are unsupported, not zero.'}
    (out/'comparison-summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['preserved_sha256','implementation_sha256']},indent=2))

if __name__=='__main__':main()
