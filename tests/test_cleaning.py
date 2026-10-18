import pandas as pd
from jack_flavours.cleaning import canonical_title, clean_appearances
from jack_flavours.features import Dataset


def test_only_confirmed_ingredient_orders_merge():
    assert canonical_title('Mascarpone & Vanilla') == canonical_title('Vanilla Mascarpone')
    assert canonical_title('Brown Sugar Vanilla') == canonical_title('Vanilla Brown Sugar')
    assert canonical_title('Pineapple & Coconut') == canonical_title('Coconut Pineapple')
    assert canonical_title('Rose Vanilla') != canonical_title('Vanilla Rose')
    assert canonical_title('Mascarpone Vanilla Brownie') != canonical_title('Vanilla Mascarpone')


def test_aliases_form_union_preserving_source_rows():
    a=pd.DataFrame({'snapshot_date':['2026-01-01']*2,'shop':['a']*2,'raw_name':['Vanilla Mascarpone','Mascarpone Vanilla'],'canonical_name':['vanilla mascarpone','mascarpone vanilla']})
    s=pd.DataFrame({'snapshot_date':['2026-01-01'],'shop':['a'],'parse_status':['valid']})
    clean=clean_appearances(a)
    assert clean.raw_name.equals(a.raw_name)
    assert clean.original_canonical_name.equals(a.canonical_name)
    assert Dataset.from_tables(s,clean,['a']).presence.sum()==1
    assert clean_appearances(a,confirmed_aliases=False).canonical_name.nunique()==2
