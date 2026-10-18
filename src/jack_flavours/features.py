"""Calendar-aligned, strictly lagged features for next-day snapshots."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['weekday_sin','weekday_cos','month_sin','month_cos','weekend','shop','lag_menu_size','lag1','lag2','lag7','lag2_missing','lag7_missing','rate7','rate30','rate90','observed7','observed30','observed90','days_since_seen','never_seen','streak']

@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    shops: list[str]
    names: list[str]
    observed: np.ndarray
    presence: np.ndarray

    @classmethod
    def load(cls, directory: Path, shops: list[str]):
        return cls.from_tables(pd.read_parquet(directory/'snapshots.parquet'),pd.read_parquet(directory/'appearances.parquet'),shops)

    @classmethod
    def from_tables(cls,snapshots,appearances,shops):
        dates=pd.date_range(snapshots.snapshot_date.min(),snapshots.snapshot_date.max())
        names=sorted(appearances.canonical_name.unique())
        di={d.date().isoformat():i for i,d in enumerate(dates)}
        si={s:i for i,s in enumerate(shops)}; fi={s:i for i,s in enumerate(names)}
        observed=np.zeros((len(dates),len(shops)),dtype=bool)
        presence=np.zeros((len(dates),len(shops),len(names)),dtype=np.uint8)
        if snapshots.duplicated(['snapshot_date','shop']).any(): raise ValueError('Duplicate snapshot')
        for row in snapshots.itertuples():
            observed[di[str(row.snapshot_date)],si[row.shop]]=row.parse_status=='valid'
        for row in appearances.itertuples():
            d,s=di[str(row.snapshot_date)],si[row.shop]
            if not observed[d,s]: raise ValueError('Appearance linked to invalid snapshot')
            presence[d,s,fi[row.canonical_name]]=1
        return cls(dates,shops,names,observed,presence)

    @property
    def eligible(self):
        prev=np.vstack([np.zeros((1,len(self.shops)),bool),self.observed[:-1]])
        return (self.observed & prev).ravel()

    @property
    def rows(self):
        return pd.DataFrame({'target_date':np.repeat(self.dates,len(self.shops)), 'shop':np.tile(self.shops,len(self.dates))})

    @property
    def y(self): return self.presence.reshape(-1,len(self.names))

    def mask(self,start=None,end=None):
        days=self.rows.target_date
        return self.eligible & (True if start is None else days.ge(pd.Timestamp(start)).to_numpy()) & (True if end is None else days.le(pd.Timestamp(end)).to_numpy())

    def vocabulary(self,train_mask,min_support):
        return np.flatnonzero(self.y[train_mask].sum(axis=0)>=min_support)

    def features(self,index):
        """Row t uses observations through t-1 only; calendar variables describe t."""
        blocks=[]
        d=len(self.dates)
        weekday=self.dates.dayofweek.to_numpy(); month=self.dates.month.to_numpy()
        for s in range(len(self.shops)):
            values=self.presence[:,s,index].astype(float)
            values[~self.observed[:,s]]=np.nan
            series=pd.Series(values)
            lag=series.shift(1)
            seen=np.full(d,np.nan); streak=np.zeros(d)
            last=None; run=0
            for t in range(d):
                # Record state BEFORE reading day t, the target day.
                if last is not None: seen[t]=t-last
                streak[t]=run
                if np.isnan(values[t]): run=0
                elif values[t]==1: last=t; run+=1
                else: run=0
            menu_size=self.presence[:,s].sum(axis=1).astype(float)
            menu_size[~self.observed[:,s]]=np.nan
            lag_size=pd.Series(menu_size).shift(1).fillna(0).to_numpy()
            rates=[lag.rolling(w,min_periods=1).mean().fillna(0).to_numpy() for w in [7,30,90]]
            counts=[lag.rolling(w,min_periods=1).count().fillna(0).to_numpy() for w in [7,30,90]]
            blocks.append(np.column_stack([np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),np.sin(2*np.pi*(month-1)/12),np.cos(2*np.pi*(month-1)/12),weekday>=5,np.full(d,s),lag_size,lag.fillna(0),series.shift(2).fillna(0),series.shift(7).fillna(0),series.shift(2).isna(),series.shift(7).isna(),*rates,*counts,np.nan_to_num(seen,nan=999),np.isnan(seen),streak]))
        return np.stack(blocks,axis=1).reshape(-1,len(FEATURES)).astype(np.float32)


def make_splits(data:Dataset,config:dict):
    end=data.dates[-1]; test_start=end-pd.Timedelta(days=config['test_days']-1)
    validation_start=test_start-pd.Timedelta(days=config['validation_folds']*config['validation_days'])
    folds=[]
    for i in range(config['validation_folds']):
        start=validation_start+pd.Timedelta(days=i*config['validation_days']); stop=start+pd.Timedelta(days=config['validation_days']-1)
        folds.append({'name':f'validation_{i+1}','train_end':str((start-pd.Timedelta(days=1)).date()),'start':str(start.date()),'end':str(stop.date())})
    if pd.Timestamp(folds[0]['train_end'])-data.dates[0]<pd.Timedelta(days=90):
        raise ValueError('Insufficient initial training history')
    return {'folds':folds,'test':{'train_end':str((test_start-pd.Timedelta(days=1)).date()),'start':str(test_start.date()),'end':str(end.date())},'policy':'Split on target date; both shops share dates. Eligible iff target and previous calendar day are valid. Test outcomes remain unused until selection is frozen.'}


def freeze(data,config,path,report):
    split=make_splits(data,config)
    if path.exists() and json.loads(path.read_text())!=split: raise ValueError('Refusing to replace a different frozen split')
    path.write_text(json.dumps(split,indent=2)+'\n')
    info=[]
    for fold in split['folds']:
        train=data.mask(end=fold['train_end']); val=data.mask(fold['start'],fold['end'])
        vocab=data.vocabulary(train,config['min_support'])
        if not train.any() or not val.any() or not len(vocab): raise ValueError('Empty training/validation fold')
        info.append({**fold,'train_forecasts':int(train.sum()),'validation_forecasts':int(val.sum()),'eligible_training_flavours':len(vocab)})
    report.write_text('# Frozen evaluation protocol\n\n```json\n'+json.dumps({'splits':split,'training_validation_inventory':info},indent=2)+'\n```\n\nSupport: at least five positive eligible training targets per flavour. All shops remain on the same side of each split. No test flavour frequencies or scores are reported here.\n\nFeatures use only issue-day and older observations. Rates use observed counts in calendar windows; unknown observations are omitted from denominators. Streaks break at gaps. Previous-day validity is required for all compared predictors. Initial training-only transforms and vocabulary are refit per fold.\n')
    return info
