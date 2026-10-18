import joblib
import numpy as np
import pandas as pd
import pytest
from jack_flavours.final import fit_bundle
from jack_flavours.forecast import build_forecast
from test_final import fixture,selection


def bundle_for(data):
    bundle=fit_bundle(data,selection(),{'seed':42})
    bundle.update(recommended='logit',operating_point={'k':1,'threshold':.5})
    return bundle


def test_saved_model_roundtrip_and_future_invariance(tmp_path):
    data=fixture();bundle=bundle_for(data)
    path=tmp_path/'model.joblib';joblib.dump(bundle,path);loaded=joblib.load(path)
    original,meta=build_forecast(loaded,data,'2025-02-15')
    data.presence[46:]=1-data.presence[46:];data.observed[47:]=False
    changed,_=build_forecast(loaded,data,'2025-02-15')
    pd.testing.assert_frame_equal(original,changed)
    assert meta['target_date']=='2025-02-16'
    assert original.top_k.all() and len(original)==2
    assert original.probability.between(0,1).all()


def test_forecast_omits_unavailable_shop_and_refuses_stale_or_leaking_dates():
    data=fixture();bundle=bundle_for(data)
    data.observed[-1,0]=False
    frame,meta=build_forecast(bundle,data)
    assert set(frame.shop)=={'b'} and meta['unavailable_shops']==['a']
    data.observed[-1,1]=False
    with pytest.raises(ValueError,match='stale'):build_forecast(bundle,data)
    with pytest.raises(ValueError,match='precedes'):build_forecast(bundle,data,'2025-01-10')
    with pytest.raises(ValueError,match='exist'):build_forecast(bundle,data,'2025-06-01')
