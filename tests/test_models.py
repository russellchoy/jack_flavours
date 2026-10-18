import numpy as np
from jack_flavours.models import fit_binary,choose_model

CONFIG={'seed':42,'boosting':{'max_iter':10,'max_leaf_nodes':3,'min_samples_leaf':2,'early_stopping':False}}


def test_single_class_fallback_both_outcomes():
    for value in [0,1]:
        for kind in ['logit','boosting']:
            model=fit_binary(np.array([[1],[2]]),np.array([value,value]),kind,CONFIG)
            np.testing.assert_array_equal(model.predict_proba([[999]])[:,1],[value])


def test_regularisation_and_training_only_scaler():
    x=np.arange(40,dtype=float).reshape(-1,1);y=(x[:,0]>20).astype(int)
    strong=fit_binary(x,y,'logit',CONFIG,0.01)
    weak=fit_binary(x,y,'logit',CONFIG,10)
    assert abs(strong[-1].coef_[0,0])<abs(weak[-1].coef_[0,0])
    assert strong[0].mean_[0]==19.5
    assert np.isfinite(strong.predict_proba([[10000]])).all()


def test_no_random_boosting_validation_split():
    x=np.arange(40,dtype=float).reshape(-1,1);y=(x[:,0]>20).astype(int)
    model=fit_binary(x,y,'boosting',CONFIG)
    assert model.early_stopping is False
    assert model.predict_proba([[30]])[0,1]>model.predict_proba([[2]])[0,1]


def entry(ap,brier):
    return {'pooled':{'micro_ap':ap,'brier':brier},'folds':[{'metrics':{'micro_ap':ap},'by_shop':{s:{'micro_ap':ap,'forecasts':30} for s in ['a','b']}} for _ in range(3)]}


def test_selection_requires_baseline_gate_and_meaningful_boosting_gain():
    summary={'frequency':entry(.2,.04),'persistence':entry(.4,.03),'logit':entry(.6,.02),'boosting':entry(.605,.019)}
    assert choose_model(summary)['selected']=='logit'
    summary['boosting']=entry(.63,.019)
    assert choose_model(summary)['selected']=='boosting'
    summary['logit']=entry(.3,.04);summary['boosting']=entry(.35,.035)
    assert choose_model(summary)['selected']=='persistence'
