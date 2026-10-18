from pathlib import Path
import sys
import numpy as np
import pandas as pd
from jack_flavours.features import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cooccurrence import cooccurrence


def test_cooccurrence_counts_conditionals_missingness_and_shop_separation():
    # Shop a: AB, A, BC, unknown. Shop b: C on all four days.
    x=np.array([[[1,1,0],[0,0,1]],[[1,0,0],[0,0,1]],[[0,1,1],[0,0,1]],[[1,1,1],[0,0,1]]],dtype=np.uint8)
    d=Dataset(pd.date_range('2026-01-01',periods=4),['a','b'],['A','B','C'],np.array([[1,1],[1,1],[1,1],[0,1]],dtype=bool),x)
    f=cooccurrence(d)
    assert len(f)==2 and f.shop.eq('a').all()
    ab=f[f.flavour_a.eq('A')].iloc[0]
    assert ab.shared_menus==1 and ab.valid_menus==3
    assert ab.b_given_a_pct==50 and ab.a_given_b_pct==50
    assert ab.lift==.75 and not ab.eligible_association
    bc=f[f.flavour_a.eq('B')].iloc[0]
    assert bc.b_given_a_pct==50 and bc.a_given_b_pct==100
    assert bc.lift==1.5


def test_common_pairs_do_not_overflow_uint8():
    d=Dataset(pd.date_range('2025-01-01',periods=300),['a'],['A','B'],np.ones((300,1),bool),np.ones((300,1,2),np.uint8))
    r=cooccurrence(d).iloc[0]
    assert r.shared_menus==300 and r.lift==1 and r.eligible_association
