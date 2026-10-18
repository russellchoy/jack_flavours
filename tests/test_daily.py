from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import pytest
from jack_flavours import daily

RAW = 'October 18th, 2026\n-\nVanilla\nChocolate\nSingle scoop £4\n'


def observation():
    return daily.validate_observation(RAW, 'all_saints', '2026-10-18', '2026-10-18T10:37:00+00:00')


def prediction():
    return {'target_date':'2026-10-18','issue_date':'2026-10-17',
            'created_utc':'2026-10-17T21:12:00+00:00', 'train_end':'2026-10-17',
            'shop':'all_saints','model':'adaptive','model_sha256':'test',
            'predictions':[{'flavour':'vanilla','probability':.8,'top_k':True},
                           {'flavour':'strawberry','probability':.1,'top_k':True}]}


def test_hand_counted_scoring_and_unknown_labels():
    result=daily.score_forecast(prediction(),observation())
    assert result['precision_at_10']==.5
    assert result['recall_at_10']==.5
    assert result['brier']==pytest.approx(.025)
    assert result['unsupported_actual']==1
    assert result['average_precision']==1


@pytest.mark.parametrize('created,cutoff', [('2026-10-18T00:00:00+00:00','2026-10-17'),
                                          ('2026-10-17T23:30:00+00:00','2026-10-17'),
                                          ('2026-10-17T12:00:00+00:00','2026-10-18')])
def test_reject_late_or_leaking_forecast(created,cutoff):
    p=prediction();p.update(created_utc=created,train_end=cutoff)
    with pytest.raises(ValueError):daily.score_forecast(p,observation())


def test_stale_and_duplicate_menus_are_not_observations():
    assert daily.validate_observation(RAW,'all_saints','2026-10-19','now')['status']!='valid'
    assert daily.validate_observation(RAW.replace('Chocolate','Vanilla'),'all_saints','2026-10-18','now')['status']!='valid'


def test_immutable_replay_and_conflict(tmp_path):
    p=tmp_path/'record.json';daily.immutable(p,{'a':1});daily.immutable(p,{'a':1})
    with pytest.raises(ValueError,match='conflict'):daily.immutable(p,{'a':2})
    assert daily.read(p)=={'a':1}


def test_retry_downloads_again_and_keeps_first_valid(tmp_path,monkeypatch):
    calls=[]
    def fetch(*args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,0,RAW if len(calls)>2 else RAW.replace('18th','17th'))
    monkeypatch.setattr(daily.subprocess,'run',fetch)
    daily.collect(tmp_path,'2026-10-18',1,'first')
    assert not (tmp_path/'observations').exists()
    daily.collect(tmp_path,'2026-10-18',2,'second')
    daily.collect(tmp_path,'2026-10-18',3,'third')
    assert len(calls)==4
    assert daily.read(tmp_path/'observations/2026-10-18/all_saints.json')['fetched_utc']=='second'


def test_score_is_idempotent_and_shared_brier_excludes_unknowns(tmp_path):
    o=observation();daily.immutable(tmp_path/'observations/2026-10-18/all_saints.json',o)
    p=prediction();daily.immutable(tmp_path/'forecasts/2026-10-18/all_saints-adaptive.json',p)
    f=deepcopy(p);f['model']='frozen';f['predictions']=f['predictions'][:1]
    daily.immutable(tmp_path/'forecasts/2026-10-18/all_saints-frozen.json',f)
    daily.score_pending(tmp_path)
    original=(tmp_path/'scores/2026-10-18/all_saints-adaptive.json').read_bytes()
    daily.score_pending(tmp_path)
    assert (tmp_path/'scores/2026-10-18/all_saints-adaptive.json').read_bytes()==original
    shared=daily.read(tmp_path/'comparisons/2026-10-18/all_saints.json')
    assert shared['shared_labels']==1
    assert shared['shared_brier']['adaptive']==pytest.approx(.04)


def test_live_history_keeps_missing_shop_and_filters_future(tmp_path):
    daily.immutable(tmp_path/'observations/2026-10-18/all_saints.json',observation())
    o=observation();o['date']='2099-01-01'
    daily.immutable(tmp_path/'observations/2099-01-01/all_saints.json',o)
    data=daily.live_dataset(tmp_path,'2026-10-18')
    assert str(data.dates[-1].date())=='2026-10-18'
    assert data.observed[-1].tolist()==[True,False]


def test_bootstrap_preserves_prior_issuance(tmp_path):
    daily.bootstrap(tmp_path);daily.bootstrap(tmp_path)
    p=daily.read(tmp_path/'forecasts/2026-10-18/all_saints-adaptive.json')
    assert p['created_utc'].startswith('2026-10-17T21:12:08')
    assert len(p['predictions'])==526


def test_max_three_attempts_and_dst_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(daily,'now_utc',lambda:datetime(2026,10,18,9,37,tzinfo=timezone.utc))
    assert daily.run(tmp_path,True)['status']=='outside_schedule'
    for i in range(3):daily.immutable(tmp_path/'runs/2026-10-18'/f'{i+1}.json',{'x':i})
    monkeypatch.setattr(daily,'now_utc',lambda:datetime(2026,10,18,10,37,tzinfo=timezone.utc))
    assert daily.run(tmp_path,True)['status']=='already_complete_or_attempt_limit'


def test_refit_failure_preserves_scores_and_reports_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(daily,'now_utc',lambda:datetime(2026,10,18,10,37,tzinfo=timezone.utc))
    def collect(state,day,attempt,timestamp):
        for shop in daily.SHOPS:
            daily.immutable(state/'observations'/day/f'{shop}.json',daily.validate_observation(RAW,shop,day,timestamp))
    monkeypatch.setattr(daily,'collect',collect)
    def fail(*args):raise RuntimeError('synthetic fit failure')
    monkeypatch.setattr(daily,'issue',fail)
    with pytest.raises(RuntimeError,match='synthetic'):daily.run(tmp_path)
    assert len(list((tmp_path/'scores/2026-10-18').glob('*.json')))==2
    assert daily.read(tmp_path/'outcomes/2026-10-18/1.json')['status']=='failed'
    assert not (tmp_path/'issued').exists()


def test_issue_then_retry_and_interrupted_recovery(tmp_path,monkeypatch):
    from test_final import fixture,selection
    data=fixture();data.shops=daily.SHOPS
    frozen=daily.fit_bundle(data,selection(),{'seed':42})
    frozen.update(recommended='logit',operating_point={'k':1,'threshold':.5})
    day=str(data.dates[-1].date())
    monkeypatch.setattr(daily,'live_dataset',lambda *args:data)
    monkeypatch.setattr(daily,'now_utc',lambda:datetime(2025,3,1,12,tzinfo=timezone.utc))
    result=daily.issue(tmp_path,day,{'seed':42},frozen,lambda:None)
    assert result['status']=='issued'
    paths=list((tmp_path/'forecasts').glob('*/*.json'))
    assert len(paths)==6
    saved={str(p):p.read_bytes() for p in paths}
    assert daily.issue(tmp_path,day,{'seed':42},frozen,lambda:None)==result
    (tmp_path/'issued'/f'{day}.json').unlink()
    paths[0].unlink()
    def no_refit(*args):raise AssertionError('Must reuse interrupted issuance')
    monkeypatch.setattr(daily,'fit_bundle',no_refit)
    recovered=daily.issue(tmp_path,day,{'seed':42},frozen,lambda:None)
    assert recovered['model_sha256']==result['model_sha256']
    assert all(Path(p).read_bytes()==v for p,v in saved.items())
