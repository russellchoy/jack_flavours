import numpy as np
import pandas as pd
from copy import deepcopy
from jack_flavours.refit import compare_forecasts
from jack_flavours.final import fit_bundle
from jack_flavours.forecast import build_forecast
from test_final import fixture,selection


def test_refit_uses_later_training_data_without_mutating_selection():
    data=fixture();original=selection();new=deepcopy(original)
    new['test_period']={'train_end':str(data.dates[-1].date())}
    bundle=fit_bundle(data,new,{'seed':42})
    assert original['test_period']['train_end']=='2025-02-09'
    assert bundle['train_forecasts']==118
    bundle.update(recommended='logit',operating_point={'k':1,'threshold':.5})
    frame,meta=build_forecast(bundle,data)
    assert meta['target_date']=='2025-03-02'
    assert set(frame.model_train_end)=={'2025-03-01'}


def test_new_support_is_missing_not_zero_and_top10_turnover():
    old=pd.DataFrame({'shop':['a','a'],'flavour':['one','two'],'probability':[.5,.4],'rank':[1,2],'top_k':[True,False]})
    new=pd.DataFrame({'shop':['a','a','a'],'flavour':['three','one','two'],'probability':[.7,.6,.3],'rank':[1,2,3],'top_k':[True,False,False]})
    compared=compare_forecasts(old,new).set_index('flavour')
    assert pd.isna(compared.loc['three','probability_old'])
    assert pd.isna(compared.loc['three','probability_change_pp'])
    assert compared.loc['three','entered_top10']
    assert compared.loc['one','left_top10']
    assert np.isclose(compared.loc['one','probability_change_pp'],10)
