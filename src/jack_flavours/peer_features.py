"""Train-only same-shop peer selection and calendar-lagged predictors."""
import numpy as np
import pandas as pd


def select_peers(data,cutoff,targets,k=5,min_support=20,min_joint=10):
    selected={int(t):{} for t in targets}; rows=[]
    for si,shop in enumerate(data.shops):
        mask=(data.dates<=pd.Timestamp(cutoff))&data.observed[:,si]
        x=data.presence[mask,si,:].astype(np.float64)
        n=len(x); counts=x.sum(axis=0); joint=x.T@x
        for t in targets:
            candidates=[]
            if counts[t]>=min_support:
                for j in np.flatnonzero((counts>=min_support)&(joint[t]>=min_joint)):
                    if j==t:continue
                    lift=float(joint[t,j]*n/(counts[t]*counts[j]))
                    if lift>1:candidates.append((int(j),lift,int(joint[t,j])))
            chosen=sorted(candidates,key=lambda v:(-v[1],-v[2],data.names[v[0]]))[:k]
            selected[int(t)][si]=[j for j,_,_ in chosen]
            for slot,(j,lift,ab) in enumerate(chosen):
                rows.append({'shop':shop,'target':data.names[t],'peer':data.names[j],'slot':slot,'lift':lift,'shared_training_menus':ab,'target_training_menus':int(counts[t]),'peer_training_menus':int(counts[j]),'valid_training_menus':n,'train_end':cutoff})
    return selected,rows


def peer_features(data,peers,k=5):
    out=np.zeros((len(data.dates),len(data.shops),len(data.shops)*k),dtype=np.float32)
    for si,indices in peers.items():
        for slot,j in enumerate(indices):
            out[1:,si,si*k+slot]=data.presence[:-1,si,j]*data.observed[:-1,si]
    return out.reshape(-1,len(data.shops)*k)
