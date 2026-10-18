"""Conservative source parsing; non-menu snapshots never become negative labels."""
from __future__ import annotations
from collections import Counter
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
import pandas as pd

FILE = re.compile(r'(?P<date>\d{2}_\d{2}_\d{2})_{1,2}(?P<shop>all_saints|benet_street)\.txt$')
MONTH = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = re.compile(MONTH + r'\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(20\d{2})', re.I)
HOLIDAYS = {'halloween': (10,31), 'all saints day':(11,1), 'guy fawkes':(11,5), 'christmas eve':(12,24), 'boxing day':(12,26), 'nye':(12,31), 'nyd':(1,1)}
STOP = re.compile(r'^(single scoop|tiny scoop|double scoop|gf cone|pint\s|thickshake|hot chocolate(?:!|\s*$)|see inside|please note|we make everything|scan the qr)', re.I)
ANNOTATION = re.compile(r'\((?:vegan|\s*\+\s*£\s*[\d.]+\s*)\)', re.I)


def normalize(name: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', ANNOTATION.sub('', name)).split()).casefold()


def parse_text(text: str, expected: date) -> tuple[str, str | None, list[dict], list[str]]:
    lines = [line.strip() for line in text.lstrip('\ufeff').splitlines()]
    found = None
    start = None
    for i, line in enumerate(lines):
        m = DATE.fullmatch(line)
        if m:
            try:
                found = datetime.strptime(f'{m[1]} {m[2]} {m[3]}', '%B %d %Y').date()
            except ValueError:
                return 'failed', None, [], ['invalid_menu_date']
            start = i + 1
            break
        h = re.fullmatch(r'(.+?),\s*(20\d{2})', line)
        if h and h[1].casefold() in HOLIDAYS:
            month, day = HOLIDAYS[h[1].casefold()]
            found = date(int(h[2]), month, day)
            start = i + 1
            break
    if found is None:
        reason = 'closure_notice' if 'closed' in text.casefold() else 'no_menu_date'
        return 'suspect', None, [], [reason]
    if found != expected:
        return 'suspect', found.isoformat(), [], ['menu_date_mismatch']
    items = []
    stopped = False
    for line in lines[start:]:
        if STOP.search(line):
            stopped = True
            break
        if not line or re.fullmatch(r'[-_–—]+', line):
            continue
        # Fail closed on an unexpected embedded header, instruction or unparenthesized price.
        remainder = ANNOTATION.sub('', line)
        if len(line) > 150 or '£' in remainder or re.search(r'\b(?:closed|opening hours|shop is|click here|https?://)\b', remainder, re.I):
            return 'suspect', found.isoformat(), [], ['unexpected_menu_line']
        canonical = normalize(line)
        if not canonical:
            return 'suspect', found.isoformat(), [], ['empty_flavour']
        items.append({'raw_name':line, 'canonical_name':canonical, 'annotations':json.dumps(ANNOTATION.findall(line),ensure_ascii=False)})
    if not stopped:
        return 'suspect', found.isoformat(), [], ['missing_footer_boundary']
    if not items:
        return 'suspect', found.isoformat(), [], ['empty_menu']
    names = [x['canonical_name'] for x in items]
    if len(names) != len(set(names)):
        return 'suspect', found.isoformat(), [], ['duplicate_flavour']
    if len(items) > 40:
        return 'suspect', found.isoformat(), [], ['implausible_menu_size']
    return 'valid', found.isoformat(), items, []


def audit(source: Path, out: Path, report: Path, config: dict, verify_source: bool = True) -> dict:
    if verify_source:
        sha = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if sha != config['source_commit']:
            raise ValueError('Source revision does not match configuration')
        if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
            raise ValueError('Source checkout must be clean')
    snapshots, appearances, bad_files = [], [], []
    keys = Counter()
    for path in sorted((source/'raw').iterdir()):
        m = FILE.fullmatch(path.name)
        if not m:
            bad_files.append(path.name)
            continue
        day = datetime.strptime(m['date'],'%y_%m_%d').date()
        raw = path.read_bytes()
        status, menu_date, items, warnings = parse_text(raw.decode('utf-8-sig'),day)
        key = (day.isoformat(),m['shop'])
        keys[key] += 1
        snapshots.append(dict(snapshot_date=key[0],shop=key[1],source_path='raw/'+path.name,source_commit=config['source_commit'],content_hash=hashlib.sha256(raw).hexdigest(),parsed_menu_date=menu_date,parse_status=status,observed_menu_size=len(items) if status=='valid' else None,warnings=';'.join(warnings)))
        appearances.extend(dict(snapshot_date=key[0],shop=key[1],**item) for item in items)
    duplicates = [key for key,count in keys.items() if count>1]
    if duplicates or bad_files:
        raise ValueError(f'Unaccounted source files or duplicate keys: {bad_files}, {duplicates}')
    if not snapshots:
        raise ValueError('No source snapshots found')
    first = min(x['snapshot_date'] for x in snapshots)
    last = max(x['snapshot_date'] for x in snapshots)
    for day in pd.date_range(first,last):
        for shop in config['shops']:
            if (day.date().isoformat(),shop) not in keys:
                snapshots.append(dict(snapshot_date=day.date().isoformat(),shop=shop,source_path=None,source_commit=config['source_commit'],content_hash=None,parsed_menu_date=None,parse_status='missing',observed_menu_size=None,warnings='missing_file'))
    s = pd.DataFrame(snapshots).sort_values(['snapshot_date','shop']).reset_index(drop=True)
    a = pd.DataFrame(appearances,columns=['snapshot_date','shop','raw_name','canonical_name','annotations']).sort_values(['snapshot_date','shop','canonical_name']).reset_index(drop=True)
    assert not s.duplicated(['snapshot_date','shop']).any()
    assert not a.duplicated(['snapshot_date','shop','canonical_name']).any()
    out.mkdir(parents=True,exist_ok=True); report.parent.mkdir(parents=True,exist_ok=True)
    s.to_parquet(out/'snapshots.parquet',index=False); a.to_parquet(out/'appearances.parquet',index=False)
    s[s.parse_status!='valid'].to_csv(out/'excluded-snapshots.csv',index=False)
    summary={'source_commit':config['source_commit'],'source_files':sum(keys.values()),'first_date':first,'last_date':last,'status_counts':s.parse_status.value_counts().to_dict(),'reason_counts':s.loc[s.parse_status!='valid','warnings'].value_counts().to_dict(),'per_shop':{shop:df.parse_status.value_counts().to_dict() for shop,df in s.groupby('shop')},'appearances':len(a),'unique_names':a.canonical_name.nunique(),'duplicates':len(duplicates),'unrecognized_files':bad_files,'menu_size_min':int(s.observed_menu_size.min()),'menu_size_max':int(s.observed_menu_size.max())}
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    report.write_text('# Data audit\n\nPinned source: `'+config['source_commit']+'`.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nAll missing, suspect and failed snapshots are excluded, not labelled empty. Full exclusions are in ignored `data/processed/excluded-snapshots.csv`. Canonical names remove only dietary/surcharge annotations, normalize Unicode/whitespace and case; no semantic alias merges. Raw names remain preserved locally.\n\nCoverage and format inspection includes the eventual test period; outcome-driven tuning does not. Historical snapshot availability is assumed after collection, not proven from commit timestamps. This is a backtest limitation.\n')
    return summary
