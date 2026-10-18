from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pytest
from jack_flavours.final import sample_calendar_blocks,bootstrap_differences,fit_bundle,predict_bundle,final_decision,check_deadline
from jack_flavours.features import Dataset


def fixture():
    dates=pd.date_range('2025-01-01',periods=60)
    observed=np.ones((60,2),bool);presence=np.zeros((60,2,1),np.uint8)
    presence[::2,:,0]=1
    return Dataset(dates,['a','b'],['vanilla'],observed,presence)


def selection():
    return {'test_period':{'train_end':'2025-02-09'},'min_support':5,'selected':'logit','logistic_C':.01,'source_commit':'synthetic'}


def test_fit_is_invariant_to_heldout_targets():
    data=fixture();first=fit_bundle(data,selection(),{'seed':42})
    x=data.features(0);pred=first['models'][0].predict_proba(x[:80])
    data.presence[40:]=1-data.presence[40:]
    second=fit_bundle(data,selection(),{'seed':42})
    np.testing.assert_allclose(second['models'][0].predict_proba(x[:80]),pred,rtol=0,atol=0)
    assert first['train_forecasts']==78


def test_predictions_match_features_and_reject_training_dates_and_gaps():
    data=fixture();bundle=fit_bundle(data,selection(),{'seed':42})
    rows=np.array([80,81]);p,indices=predict_bundle(bundle,data,rows)
    np.testing.assert_allclose(p['logit'],bundle['models'][0].predict_proba(data.features(0)[rows])[:,1,None])
    with pytest.raises(ValueError,match='cutoff'):predict_bundle(bundle,data,[0,1])
    data.observed[39,0]=False
    with pytest.raises(ValueError,match='previous'):predict_bundle(bundle,data,rows)


def test_calendar_blocks_keep_shops_together():
    dates=np.repeat(pd.date_range('2025-01-01',periods=14).to_numpy(),2)
    rows=sample_calendar_blocks(dates,'2025-01-01','2025-01-14',7,np.random.default_rng(42))
    assert len(rows)==28
    np.testing.assert_array_equal(rows[::2]+1,rows[1::2])
    # A calendar day missing both shops remains a gap, not a compressed observation day.
    partial=np.delete(dates,[6,7])
    a=sample_calendar_blocks(partial,'2025-01-01','2025-01-14',7,np.random.default_rng(42))
    assert np.isin(partial[a],dates).all()


def test_paired_bootstrap_identical_predictions_zero_difference():
    dates=np.repeat(pd.date_range('2025-01-01',periods=14).to_numpy(),2)
    y=np.tile([[1,0],[0,1]],(14,1));p=y*.6+.2
    results=bootstrap_differences(y,{'logit':p,'frequency':p,'persistence':p},dates,'logit','2025-01-01','2025-01-14',7,10,42)
    assert results['persistence']['ap_gain']['ci95']==[0,0]
    assert results['frequency']['brier_reduction']['ci95']==[0,0]


def test_predeclared_failure_falls_back_without_tuning():
    r={'logit':{'micro_ap':.4,'brier':.1},'persistence':{'micro_ap':.5,'brier':.08},'frequency':{'micro_ap':.2,'brier':.2}}
    assert final_decision(r,'logit','persistence',{})['recommended']=='persistence'
    with pytest.raises(TimeoutError):check_deadline({'deadline_utc':'2000-01-01T00:00:00+00:00'})
