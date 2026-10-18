import numpy as np
import pandas as pd
from jack_flavours.features import Dataset
from jack_flavours.peer_features import select_peers,peer_features


def example():
    x=np.zeros((8,2,3),dtype=np.uint8)
    x[:2,0,:2]=1
    x[2:4,0,2]=1
    x[:4,1,2]=1
    return Dataset(pd.date_range('2026-01-01',periods=8),['a','b'],['target','peer','other'],np.ones((8,2),bool),x)


def test_selection_excludes_self_and_future_and_separates_shops():
    d=example();a,rows=select_peers(d,'2026-01-04',[0],min_support=2,min_joint=2)
    assert a=={0:{0:[1],1:[]}}
    assert rows[0]['lift']==2
    d.presence[4:]=1
    b,_=select_peers(d,'2026-01-04',[0],min_support=2,min_joint=2)
    assert b==a


def test_features_only_use_previous_calendar_day_and_correct_shop():
    d=example();a,_=select_peers(d,'2026-01-04',[0],min_support=2,min_joint=2)
    before=peer_features(d,a[0]).reshape(8,2,10)
    assert before[0].sum()==0
    assert before[1,0,0]==1 and before[:,1].sum()==0
    d.presence[2:]=1
    after=peer_features(d,a[0]).reshape(8,2,10)
    np.testing.assert_array_equal(before[:3],after[:3])
    d.observed[1,0]=False
    assert peer_features(d,a[0]).reshape(8,2,10)[2,0].sum()==0
