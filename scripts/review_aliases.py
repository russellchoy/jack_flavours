"""Prepare a manual alias shortlist; never apply proposed merges."""
from collections import Counter
import hashlib,json,re
from pathlib import Path
import pandas as pd
from jack_flavours.live_cleaning import canonical_title_v4

source=Path('data/processed/appearances.parquet')
a=pd.read_parquet(source)
a['review_name']=a.raw_name.map(canonical_title_v4)
unique=a[['snapshot_date','shop','review_name']].drop_duplicates()
counts=unique.review_name.value_counts().to_dict()
groups={}
for name in sorted(counts):
    # Ingredient reordering is candidate generation only, never normalization.
    tokens=tuple(sorted(re.findall(r'\w+',name)))
    groups.setdefault(tokens,[]).append(name)
proposals=[{'names':names,'reason':'Same words in a different order / punctuation','category':'word order'} for names in groups.values() if len(names)>1]
for names,reason in [
    (['strawberries cream','strawberry cream'],'Singular/plural strawberry'),
    (['blackcurrant cream','blackcurrants cream'],'Singular/plural blackcurrant'),
    (['alphonso mango sorbet','alphoso mango sorbet'],'Possible missing n in Alphonso'),
    (['mint stracciatella','mint straccitella'],'Possible missing a in stracciatella'),
    (['cardamom rose','cardamon rose'],'Cardamom/cardamon spelling'),
    (['raspberry','raspberry raspberry'],'Repeated word; could also be intentional emphasis')]:
    if all(n in counts for n in names):proposals.append({'names':names,'reason':reason,'category':'spelling or wording'})
proposals.sort(key=lambda p:-sum(counts[n] for n in p['names']))
rows=[]
for i,p in enumerate(proposals,1):
    keys=[set(map(tuple,unique[unique.review_name==n][['snapshot_date','shop']].to_numpy())) for n in p['names']]
    co=sum(len(keys[x]&keys[y]) for x in range(len(keys)) for y in range(x+1,len(keys)))
    entries=[]
    for n in p['names']:
        raw=a[a.review_name==n].sort_values(['snapshot_date','shop'])
        entries.append({'name':n,'count':counts[n],'first':str(raw.snapshot_date.min()),'last':str(raw.snapshot_date.max()),'raw_examples':raw.raw_name.drop_duplicates().head(3).tolist(), 'source_example': {'date':str(raw.iloc[0].snapshot_date),'shop':raw.iloc[0].shop,'raw_name':raw.iloc[0].raw_name}})
    rows.append({'id':f'C{i:02d}',**p,'variants':entries,'same_menu_pairs':co,'decision':'pending','suggested_canonical':max(p['names'],key=lambda n:counts[n])})
report=Path('reports/alias-review.md')
lines=['# Flavour alias candidates for manual approval','','Status: proposals only. No candidate in this report has been merged. The separately approved house vanilla → vanilla rule is already handled by live version v4.','','Counts are unique shop/date menu appearances through **17 October 2026**, combining both shops; they are not serving or sales counts. Names below already use v3 cleaning plus the approved house-vanilla rule. Original raw names and dated source examples are retained in the companion JSON.','','This reuses the existing Cleanlab audit approach: ingredient-token collisions and a curated set of spelling candidates. It does not rerun Cleanlab, and similarity is not proof of recipe identity. Same-menu pairs would be a reason to investigate; zero overlap is not proof of equivalence.','','Reply with IDs, for example **approve C01, C03, C08; reject C02; leave the rest pending**. You can also change a proposed canonical name.','','| ID | Candidate variants (appearances) | Reason | Same-menu pairs |','|---|---|---|---:|']
for r in rows:
    variants=' ↔ '.join(f"{v['name']} ({v['count']})" for v in r['variants'])
    lines.append(f"| {r['id']} | {variants} | {r['reason']} | {r['same_menu_pairs']} |")
lines += ['', '## Keep separate unless you explicitly confirm otherwise','', 'Similar names that add an ingredient or change the product type are not included in the merge shortlist: vanilla versus vanilla rose; strawberry versus strawberry sorbet; house yoghurt versus house lemon yoghurt; honeycomb versus hazelnut honeycomb. No general ingredient sorting, removal of “sorbet”, or removal of “house” is proposed.', '', '## Provenance','',f"Input: data/processed/appearances.parquet; SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`. Reproduce with `uv run --locked python scripts/review_aliases.py`. Machine-readable decisions and raw examples: [alias-review.json](alias-review.json)."]
report.write_text('\n'.join(lines)+'\n')
Path('reports/alias-review.json').write_text(json.dumps({'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cutoff':'2026-10-17','normalization':'v4','proposals':rows},indent=2,ensure_ascii=False)+'\n')
print('\n'.join(lines[:12]+lines[12:24]))
print('Total candidate groups:',len(rows))
