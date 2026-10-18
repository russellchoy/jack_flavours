"""Local, read-only Cleanlab audit. Run with Documents/Cleanlab/.venv/bin/python."""
from pathlib import Path
import argparse, contextlib, hashlib, io, json, re, unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import cleanlab
from cleanlab import Datalab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def simple(s):
    s = unicodedata.normalize('NFKD', s.casefold())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    paths = [root/'data/processed/snapshots.parquet',root/'data/processed/appearances.parquet',root/'data/cleaned-descriptive-v2/appearances.parquet']
    hashes={str(p.relative_to(root)):digest(p) for p in paths}
    snapshots=pd.read_parquet(paths[0]); original=pd.read_parquet(paths[1]); a=pd.read_parquet(paths[2])
    appearances=a.drop_duplicates(['snapshot_date','shop','canonical_name'])
    counts=appearances.canonical_name.value_counts()
    names=sorted(counts.index)
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),lowercase=False)
    features=vectorizer.fit_transform(names).toarray()
    lab=Datalab(data={'name':names},verbosity=0)
    lab.find_issues(features=features,issue_types={'near_duplicate':{'metric':'cosine'},'outlier':{'metric':'cosine'}})
    issues=lab.get_issues().copy(); issues.insert(0,'name',names); issues['appearances']=issues.name.map(counts)
    issues.to_csv(out/'cleanlab-name-issues.csv',index=False)
    lab.get_issue_summary().to_csv(out/'cleanlab-name-summary.csv',index=False)
    with (out/'cleanlab-name-report.txt').open('w') as f, contextlib.redirect_stdout(f): lab.report()
    near=lab.get_info('near_duplicate')['near_duplicate_sets']
    flagged={tuple(sorted((i,int(j)))) for i,js in enumerate(near) for j in js if i!=j}
    distances,neighbors=NearestNeighbors(n_neighbors=6,metric='cosine').fit(features).kneighbors(features)
    candidates={}
    def add(i,j,reason,similarity=None):
        if i==j:return
        i,j=sorted((i,j)); row=candidates.setdefault((i,j),{'name_a':names[i],'name_b':names[j],'count_a':int(counts[names[i]]),'count_b':int(counts[names[j]]),'reasons':set(),'cleanlab_flag':(i,j) in flagged})
        row['reasons'].add(reason)
        if similarity is not None:row['cosine_similarity']=float(similarity)
    for i,j in flagged:add(i,j,'cleanlab_default_near_duplicate')
    for i,(ds,js) in enumerate(zip(distances,neighbors)):
        for d,j in zip(ds,js):
            if i!=j and (1-d>=.80 or SequenceMatcher(None,names[i],names[j]).ratio()>=.92):add(i,int(j),'similar_spelling_review',1-d)
    for key,reason in [(simple,'punctuation_accent_collision'),(lambda s:' '.join(sorted(simple(s).split())),'token_order_collision')]:
        groups=defaultdict(list)
        for i,n in enumerate(names):groups[key(n)].append(i)
        for ids in groups.values():
            for pos,i in enumerate(ids):
                for j in ids[pos+1:]:add(i,j,reason)
    result=[]
    for (i,j),row in candidates.items():
        left=set(zip(appearances.loc[appearances.canonical_name.eq(names[i]),'snapshot_date'],appearances.loc[appearances.canonical_name.eq(names[i]),'shop']))
        right=set(zip(appearances.loc[appearances.canonical_name.eq(names[j]),'snapshot_date'],appearances.loc[appearances.canonical_name.eq(names[j]),'shop']))
        row['same_menu_cooccurrences']=len(left&right)
        row['reasons']=';'.join(sorted(row['reasons']));row['total_appearances']=row['count_a']+row['count_b'];result.append(row)
    pairs=pd.DataFrame(result).sort_values(['cleanlab_flag','total_appearances'],ascending=False)
    pairs.to_csv(out/'name-candidates.csv',index=False)
    suspect=set(pairs.name_a)|set(pairs.name_b)
    a[a.canonical_name.isin(suspect)].merge(snapshots[['snapshot_date','shop','source_path']],on=['snapshot_date','shop']).to_csv(out/'candidate-source-records.csv',index=False)
    # Audit menu composition separately by shop. Recurrence is expected, so no duplicate deletion.
    menus=appearances.groupby(['snapshot_date','shop']).canonical_name.agg(lambda v:sorted(set(v))).reset_index(name='names')
    menu_issues=[]
    for shop,g in menus.groupby('shop'):
        g=g.reset_index(drop=True)
        matrix=np.zeros((len(g),len(names)),dtype=np.float64); index={n:i for i,n in enumerate(names)}
        for i,ns in enumerate(g.names):matrix[i,[index[n] for n in ns]]=1
        audit=Datalab(data={'date':g.snapshot_date.tolist()},verbosity=0)
        audit.find_issues(features=matrix,issue_types={'outlier':{'metric':'cosine'}})
        f=audit.get_issues().copy();f.insert(0,'shop',shop);f.insert(0,'date',g.snapshot_date);f['menu_size']=matrix.sum(axis=1).astype(int);menu_issues.append(f)
    pd.concat(menu_issues).to_csv(out/'cleanlab-menu-issues.csv',index=False)
    signatures=defaultdict(list)
    for r in menus.itertuples():signatures[(r.snapshot_date,tuple(r.names))].append(r.shop)
    identical=[{'date':d,'shops':ss,'menu_size':len(ns),'names':list(ns)} for (d,ns),ss in signatures.items() if len(ss)>1]
    (out/'identical-cross-shop-menus.json').write_text(json.dumps(identical,indent=2))
    schema={'snapshot_status_counts':snapshots.parse_status.value_counts().to_dict(),'duplicate_snapshot_keys':int(snapshots.duplicated(['snapshot_date','shop']).sum()),'original_duplicate_appearance_keys':int(original.duplicated(['snapshot_date','shop','canonical_name']).sum()),'cleaned_duplicate_appearance_keys':int(a.duplicated(['snapshot_date','shop','canonical_name']).sum()),'missing_raw_names':int(a.raw_name.isna().sum()),'valid_menu_size_min':int(menus.names.str.len().min()),'valid_menu_size_max':int(menus.names.str.len().max())}
    assert all(digest(p)==hashes[str(p.relative_to(root))] for p in paths)
    summary={'cleanlab_version':cleanlab.__version__,'input_sha256':hashes,'unique_names':len(names),'valid_menus':len(menus),'feature_dimensions':features.shape[1],'feature_method':'character word-boundary TF-IDF 2–4 grams; cosine distance; Cleanlab default thresholds/k=10','cleanlab_name_summary':lab.get_issue_summary().to_dict('records'),'cleanlab_flagged_pairs':len(flagged),'all_review_candidate_pairs':len(pairs),'cleanlab_menu_outliers':int(pd.concat(menu_issues).is_outlier_issue.sum()),'identical_same_date_cross_shop_menus':len(identical),'schema':schema,'label_error_detection':'Not run: no independent true recipe labels or suitable out-of-sample probabilities for name identity. Forecast surprise is not evidence of a scraping error.','inputs_unchanged':True}
    (out/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();main(args.root,args.out)
