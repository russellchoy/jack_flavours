import numpy as np
import pytest
from jack_flavours.evaluation import metrics,pooled,choose_operating_point


def test_ties_novelty_and_no_positive_labels():
    y=np.array([[1,0,0],[0,1,0]])
    p=np.full((2,3),0.5)
    result=metrics(y,p,np.array([2,1]),k=1)
    assert result['precision_at_k']==0.5
    assert result['recall_at_k_all_flavours']==0.25
    assert result['positive_labels_for_macro']==2
    assert result['vocabulary_positive_coverage']==2/3


def test_zero_positive_and_bad_probabilities():
    result=metrics(np.zeros((2,2)),np.zeros((2,2)),k=1)
    assert result['micro_ap'] is None and result['macro_ap'] is None
    assert result['brier']==0 and result['f1']==0
    with pytest.raises(ValueError):metrics(np.zeros((1,1)),np.array([[1.1]]))


def test_pool_and_operating_point_with_different_vocabularies():
    records=[{'y':np.array([[1,0]]),'m':np.array([[0.8,0.2]]),'total':np.array([1])},{'y':np.array([[0,1,0]]),'m':np.array([[0.1,0.9,0.2]]),'total':np.array([1])}]
    assert pooled(records,'m')['micro_ap']==1
    assert choose_operating_point(records,'m',[1,2],[0.2,0.5])=={'threshold':0.5,'k':1}
