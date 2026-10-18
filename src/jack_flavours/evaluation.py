"""Paired forecast metrics with explicit coverage and deterministic ranking ties."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def metrics(y,p,total_positives=None,k=12,threshold=0.5):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or y.ndim!=2 or not y.size: raise ValueError('Nonempty matching matrices required')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid probabilities')
    positive=y.sum(axis=0)>0
    micro=float(average_precision_score(y.ravel(),p.ravel())) if y.sum() else None
    macro=float(np.mean([average_precision_score(y[:,i],p[:,i]) for i in np.flatnonzero(positive)])) if positive.any() else None
    precision,recall,f1,_=precision_recall_fscore_support(y.ravel(),(p>=threshold).ravel(),average='binary',zero_division=0)
    k=min(k,y.shape[1])
    if k<1: raise ValueError('k must be positive')
    # Columns are lexically ordered names; stable sort resolves ties lexically.
    top=np.argsort(-p,axis=1,kind='stable')[:,:k]
    hits=np.take_along_axis(y,top,axis=1).sum(axis=1)
    actual=y.sum(axis=1) if total_positives is None else np.asarray(total_positives)
    recall_k=np.divide(hits,actual,out=np.zeros_like(hits,dtype=float),where=actual>0)
    precision_k=hits/k
    f1_k=np.divide(2*precision_k*recall_k,precision_k+recall_k,out=np.zeros(len(y)),where=(precision_k+recall_k)>0)
    bins=[]
    flat_p=p.ravel(); flat_y=y.ravel()
    for lo in np.arange(10)/10:
        mask=(flat_p>=lo)&((flat_p<lo+0.1) if lo<0.9 else (flat_p<=1))
        bins.append({'lower':float(lo),'count':int(mask.sum()),'mean_probability':float(flat_p[mask].mean()) if mask.any() else None,'observed_rate':float(flat_y[mask].mean()) if mask.any() else None})
    return {'forecasts':len(y),'label_count':y.shape[1],'label_forecast_pairs':int(y.size),'positive_labels_for_macro':int(positive.sum()),'micro_ap':micro,'macro_ap':macro,'brier':float(np.mean((p-y)**2)),'precision':float(precision),'recall':float(recall),'f1':float(f1),'threshold':threshold,'k':k,'precision_at_k':float(precision_k.mean()),'recall_at_k_all_flavours':float(recall_k.mean()),'menu_f1_at_k':float(f1_k.mean()),'in_vocabulary_positives':int(y.sum()),'all_flavour_positives':int(actual.sum()),'vocabulary_positive_coverage':float(y.sum()/actual.sum()) if actual.sum() else None,'empty_target_menus':int((actual==0).sum()),'calibration':bins}


def pooled(records,model):
    """Pool comparable out-of-fold pairs even when training vocabularies grow."""
    ys=np.concatenate([r['y'].ravel() for r in records]); ps=np.concatenate([r[model].ravel() for r in records])
    return {'micro_ap':float(average_precision_score(ys,ps)) if ys.sum() else None,'brier':float(np.mean((ys-ps)**2)),'label_forecast_pairs':len(ys)}


def choose_operating_point(records,model,ks,thresholds):
    y=np.concatenate([r['y'].ravel() for r in records]);p=np.concatenate([r[model].ravel() for r in records])
    # Prefer higher threshold when F1 ties; smaller menu when menu F1 ties.
    scored=[]
    for threshold in thresholds:
        _,_,f1,_=precision_recall_fscore_support(y,p>=threshold,average='binary',zero_division=0)
        scored.append((float(f1),threshold))
    threshold=max(scored)[1]
    scored_k=[]
    for k in ks:
        score=sum(metrics(r['y'],r[model],r['total'],k,threshold)['menu_f1_at_k']*len(r['y']) for r in records)/sum(len(r['y']) for r in records)
        scored_k.append((score,-k))
    return {'threshold':threshold,'k':-max(scored_k)[1]}
