"""Descriptive completed menu/absence spells, with explicit censoring at gaps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from jack_flavours.features import Dataset


def spells(state):
    """Return known-start spells. -1 means unknown, never an absence."""
    result=[]
    start=0
    while start<len(state):
        end=start+1
        while end<len(state) and state[end]==state[start]:end+=1
        if state[start]>=0 and start>0 and state[start-1]>=0:
            # A true observed state transition anchors the beginning.
            result.append({'state':int(state[start]),'start':start,'days':end-start,'completed':end<len(state) and state[end]>=0,'end_reason':'transition' if end<len(state) and state[end]>=0 else ('end_of_history' if end==len(state) else 'unknown_snapshot')})
        start=end
    return result


def stats(df):
    completed=df.loc[df.completed,'days']
    return {'completed_spells':len(completed),'mean_days':float(completed.mean()) if len(completed) else None,'median_days':float(completed.median()) if len(completed) else None,'p25_days':float(completed.quantile(.25)) if len(completed) else None,'p75_days':float(completed.quantile(.75)) if len(completed) else None,'p90_days':float(completed.quantile(.9)) if len(completed) else None,'one_day_pct':float((completed==1).mean()*100) if len(completed) else None,'incomplete_known_start_spells':int((~df.completed).sum()),'censored_at_gap':int((df.end_reason=='unknown_snapshot').sum()),'censored_at_history_end':int((df.end_reason=='end_of_history').sum())}


def main():
    # Independent hand-calculated examples: complete spells, gaps and boundary censoring.
    assert spells([0,1,1,0,0,1,0])==[
        {'state':1,'start':1,'days':2,'completed':True,'end_reason':'transition'},
        {'state':0,'start':3,'days':2,'completed':True,'end_reason':'transition'},
        {'state':1,'start':5,'days':1,'completed':True,'end_reason':'transition'},
        {'state':0,'start':6,'days':1,'completed':False,'end_reason':'end_of_history'}]
    gaps=spells([1,0,-1,0,1,1,-1])
    assert len(gaps)==2 and not any(x['completed'] for x in gaps)
    assert spells([1,1,0])[0]['state']==0 # Unknown initial start excluded.
    config=json.loads(Path('configs/experiment.json').read_text())
    data=Dataset.load(Path('data/processed'),config['shops'])
    rows=[]
    for s,shop in enumerate(data.shops):
        for f,name in enumerate(data.names):
            values=data.presence[:,s,f].astype(np.int8)
            values[~data.observed[:,s]]=-1
            for item in spells(values):
                rows.append({'shop':shop,'flavour':name,'start_date':str(data.dates[item['start']].date()),**item})
    frame=pd.DataFrame(rows)
    output={}
    for shop,df in [('both_shops',frame),*list(frame.groupby('shop'))]:
        output[shop]={key:stats(df[df.state==state]) for key,state in [('on_menu',1),('off_menu_until_return',0)]}
    # A wider, descriptive observed-return count is separate from confirmed absence lengths.
    returns=[]
    for s,shop in enumerate(data.shops):
        for f,name in enumerate(data.names):
            seen=np.flatnonzero(data.presence[:,s,f])
            for a,b in zip(seen[:-1],seen[1:]):
                if b-a>1 and data.observed[a+1,s] and not data.presence[a+1,s,f]:
                    returns.append({'shop':shop,'days':int(b-a-1),'continuous':bool(data.observed[a+1:b,s].all())})
    return_frame=pd.DataFrame(returns)
    output['observed_returns_including_intervening_unknown_days']={shop:{'count':len(df),'mean_calendar_days':float(df.days.mean()),'median_calendar_days':float(df.days.median()),'with_unknown_days':int((~df.continuous).sum())} for shop,df in [('both_shops',return_frame),*list(return_frame.groupby('shop'))]}
    out=Path('artifacts/menu-durations');out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'spells.csv',index=False)
    (out/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Menu stay and return durations','','Descriptive history: 4 October 2024–17 October 2026, pinned source '+config['source_commit']+'. No model fitting or retuning.','', '| Shop | On-menu mean | On-menu median | Completed stays | Off-menu mean | Off-menu median | Completed return gaps |','|---|---:|---:|---:|---:|---:|---:|']
    for shop in ['both_shops',*data.shops]:
        a=output[shop]['on_menu'];b=output[shop]['off_menu_until_return'];lines.append(f"| {shop} | {a['mean_days']:.2f} | {a['median_days']:.0f} | {a['completed_spells']} | {b['mean_days']:.2f} | {b['median_days']:.0f} | {b['completed_spells']} |")
    lines+=['','Each observation is a flavour at one shop. Means are weighted by completed spells, not equally by flavour. A completed stay begins after an observed absent day and ends at the next observed absent day. A completed return gap begins after a present day and ends at the next present day. Daily snapshots are not continuous stock observations.','', 'Only uninterrupted, valid daily observations with both transitions confirmed enter these means. Missing, suspect or failed snapshots censor a spell; the history boundary also censors it. Spells already in progress at the start of a valid segment are excluded. A one-day absence means absent in one daily snapshot before returning the following day.','', 'These are conditional completed-spell averages, NOT unconditional expected wait times. Long stays/gaps are more likely to cross a missing day or the history endpoint and be excluded. Flavours that never return do not enter the completed-return mean. Conservative name matching also treats unmerged spelling/name changes as different flavours. Do not infer a guaranteed return schedule from these figures.','', 'A separate sensitivity summary includes next observed appearances across unknown days: that is elapsed time until next observed return, not proven continuous absence, because unobserved intermediate returns are possible.','', '```json',json.dumps(output,indent=2),'```','', 'Reproduce with `.venv/bin/python scripts/menu_durations.py`. Three hand-calculated synthetic histories verify complete spells, missing-day censoring and unknown initial starts. Full spell-level records remain local in artifacts/menu-durations/spells.csv.']
    Path('reports/menu-durations.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
