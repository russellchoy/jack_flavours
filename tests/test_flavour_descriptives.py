"""Hand-counted recurrence examples for the descriptive charts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from jack_flavours.features import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from flavour_descriptives import summarise


def test_completed_durations_and_recorded_returns_are_distinct():
    states = np.array([0,1,1,0,0,1,0,-1,1,0,1])
    data = Dataset(pd.date_range('2026-01-01', periods=len(states)), ['shop'], ['vanilla'], (states>=0)[:,None], (states==1).astype(np.uint8)[:,None,None])
    frame, events = summarise(data)
    row = frame.iloc[0]
    assert row.appearances == 5
    assert row.valid_menus == 10
    assert row.stay_n == 2  # two days, one day; later starts/ends censored
    assert row.stay_mean == 1.5
    assert row.absence_n == 2
    assert row.absence_mean == 1.5  # two-day and one-day fully observed gaps
    assert row.recorded_return_n == 3
    assert row.recorded_return_median == 2
    assert row.between_snapshots_mean == 2.25  # 1, 3, 3, 2 elapsed days
    assert len(events[(events.kind=='recorded_return') & ~events.complete]) == 1


def test_menu_similarity_uses_union_and_preserves_unknown_dates():
    from flavour_descriptives import menu_similarity
    data = Dataset(pd.date_range('2026-01-01', periods=5), ['a','b'], ['x','y','z'],
                   np.array([[1,1],[1,1],[1,1],[1,0],[1,1]], dtype=bool),
                   np.array([[[1,1,0],[0,1,1]], [[1,0,0],[1,0,0]],
                             [[1,0,0],[0,1,0]], [[1,0,0],[0,0,0]],
                             [[0,0,0],[0,0,0]]], dtype=np.uint8))
    result = menu_similarity(data)
    np.testing.assert_allclose(result.similarity_pct[:3], [100/3,100,0])
    assert result.similarity_pct[3:].isna().all()
    assert pd.isna(result.shared_flavours[3])
    assert result.distinct_flavours[0] == 3


def test_conjunction_matching_preserves_recipes_and_deduplicates_menus():
    from jack_flavours.cleaning import normalize_title, clean_appearances
    assert normalize_title('Coconut & Ube (vegan)') == normalize_title('Coconut and Ube') == normalize_title('coconut ube')
    assert normalize_title('Pandan & Candy') == 'pandan candy'
    assert normalize_title('Coconut') != normalize_title('Ube')
    assert normalize_title('Coconut ube, rose') != normalize_title('Coconut ube')
    original = pd.DataFrame({'snapshot_date': ['2026-01-01']*2, 'shop': ['a']*2,
                             'raw_name': ['Coconut & Ube', 'Coconut Ube'],
                             'canonical_name': ['coconut & ube', 'coconut ube']})
    clean = clean_appearances(original)
    assert clean.raw_name.equals(original.raw_name)
    assert original.canonical_name.nunique() == 2
    snapshots = pd.DataFrame({'snapshot_date':['2026-01-01'], 'shop':['a'], 'parse_status':['valid']})
    data = Dataset.from_tables(snapshots, clean, ['a'])
    assert data.presence.sum() == 1


def test_autocorrelogram_uses_calendar_lags_and_valid_endpoints():
    from flavour_descriptives import menu_autocorrelogram
    data = Dataset(pd.date_range('2026-01-01', periods=4), ['a'], ['x','y','z'],
                   np.array([[1],[1],[0],[1]], dtype=bool),
                   np.array([[[1,1,0]], [[0,1,1]], [[0,0,0]], [[1,1,0]]], dtype=np.uint8))
    result = menu_autocorrelogram(data, 10)
    assert result.lag_days.tolist() == [0,1,2,3]
    assert result.pairs.tolist() == [3,1,1,1]
    np.testing.assert_allclose(result.similarity_pct, [100,100/3,100/3,100])
    data.observed[:] = False
    result = menu_autocorrelogram(data, 1)
    assert result.pairs.eq(0).all()
    assert result.similarity_pct.isna().all()
