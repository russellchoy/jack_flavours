import numpy as np
import pandas as pd
from jack_flavours.features import Dataset
from scripts.novelty_experiment import novelty_data,status_features,prequential,novelty_eligible


def fixture():
    dates=pd.date_range('2025-01-01',periods=180)
    presence=np.zeros((180,2,2),np.uint8)
    presence[1:4,0,0]=1;presence[7:9,0,0]=1;presence[10,1,0]=1
    presence[30,:,1]=1
    return Dataset(dates,['a','b'],['one','two'],np.ones((180,2),bool),presence)


def test_global_vs_local_novelty_and_simultaneous_introduction():
    data=fixture();x,y,count,local=novelty_data(data)
    assert count[2]==1 and count[4]==0
    assert count[21]==0 and local[21]==1
    assert count[60]==count[61]==1
    assert not novelty_eligible(data,90)[:180].any()


def test_features_cannot_see_future_introductions_or_returns():
    data=fixture();x,_,_,_=novelty_data(data);status=status_features(data,0)
    assert status[16,1]>0 and status[4,1]==0
    data.presence[20:]=1-data.presence[20:]
    after,_,_,_=novelty_data(data)
    np.testing.assert_array_equal(x[:42],after[:42])
    np.testing.assert_array_equal(status[:42],status_features(data,0)[:42])


def test_crossfit_has_strict_cutoffs_and_future_invariance():
    data=fixture();x,y,_,_=novelty_data(data)
    config={'burn_in_days':30,'inner_block_days':30,'min_novelty_training_rows':10,'logistic_C':.01,'seed':42}
    p,cut=prequential(x,y,data,config)
    assert (cut<np.repeat(np.arange(180),2)).all()
    y[240:]=1-y[240:];x[242:]+=17
    future,_=prequential(x,y,data,config)
    np.testing.assert_array_equal(p[:242],future[:242])


def test_missing_observation_does_not_create_a_return():
    data=fixture();data.observed[4:7,0]=False
    assert status_features(data,0)[16,1]==0
