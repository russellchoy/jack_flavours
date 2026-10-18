import numpy as np
import pandas as pd
from jack_flavours.features import Dataset,FEATURES,make_splits


def fixture():
    days=pd.date_range('2025-01-01',periods=220)
    snaps=[{'snapshot_date':str(d.date()),'shop':s,'parse_status':'missing' if i==5 else 'valid'} for i,d in enumerate(days) for s in ['a','b']]
    apps=[{'snapshot_date':str(d.date()),'shop':s,'canonical_name':'vanilla'} for i,d in enumerate(days) for s in ['a','b'] if i!=5 and i%3==0]
    apps.append({'snapshot_date':str(days[-1].date()),'shop':'a','canonical_name':'new'})
    return Dataset.from_tables(pd.DataFrame(snaps),pd.DataFrame(apps),['a','b'])


def test_future_mutation_cannot_change_features_at_cutoff():
    data=fixture(); f=data.names.index('vanilla'); before=data.features(f).copy()
    data.presence[20:,:,:]=1-data.presence[20:,:,:]
    data.observed[21:]=False
    after=data.features(f)
    # Includes forecast day 20: it may use day 19, never day 20.
    np.testing.assert_array_equal(before[:42],after[:42])


def test_missing_days_not_negative_or_bridged():
    data=fixture(); f=data.names.index('vanilla'); x=data.features(f)
    assert not data.eligible[10:14].any()  # target missing day, then issue missing day
    assert x[12,FEATURES.index('lag1')]==0
    assert x[12,FEATURES.index('observed7')]==5
    assert x[12,FEATURES.index('rate7')]==2/5 or np.isclose(x[12,FEATURES.index('rate7')],2/5)
    assert x[12,FEATURES.index('streak')]==0


def test_training_vocabulary_excludes_future_label():
    data=fixture(); train=data.mask(end='2025-04-01')
    assert [data.names[i] for i in data.vocabulary(train,5)]==['vanilla']


def test_shared_nonoverlapping_target_date_splits():
    data=fixture(); split=make_splits(data,{'test_days':30,'validation_folds':3,'validation_days':20})
    for fold in split['folds']:
        train=data.mask(end=fold['train_end']); val=data.mask(fold['start'],fold['end'])
        assert not (train&val).any()
        assert data.rows[train].target_date.max()<data.rows[val].target_date.min()
        for _,rows in data.rows[val].groupby('target_date'): assert len(rows)==2
