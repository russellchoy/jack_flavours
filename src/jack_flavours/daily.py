"""Append-only prospective daily forecasting. All dates use Europe/London."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .cleaning import canonical_title, clean_appearances
from .live_cleaning import canonical_title_v4, canonical_title_v5
from .data import parse_text
from .features import Dataset
from .final import digest, fit_bundle
from .forecast import build_forecast

LONDON = ZoneInfo('Europe/London')
SCRAPER = 'ccdfca95d4eef554052ebab63d0c9733bc96624f'
SHOPS = ['all_saints', 'benet_street']
DOCS = {'all_saints': '1kDBSxPb8X4L2TKXWUmm2A-VGuPVTyxmfbq9iwUQQ2nc',
        'benet_street': '1dVYB7lnBgWE0bPhc9SFz0aLrkDfSCulrMctW1gDfCA8'}
START = '2026-10-18'
NORMALIZATION = 'v5'


def now_utc():
    return datetime.now(timezone.utc)


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    """An existing record may be replayed, but never silently revised."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable record conflict: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def validate_observation(raw, shop, day, fetched):
    status, menu_date, items, warnings = parse_text(raw, datetime.fromisoformat(day).date())
    return {'schema': 1, 'date': day, 'shop': shop, 'fetched_utc': fetched,
            'document_id': DOCS[shop], 'scraper_commit': SCRAPER,
            'normalization': 'original-v1', 'raw': raw,
            'sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'menu_date': menu_date, 'status': status, 'warnings': warnings,
            'flavours': sorted({i['canonical_name'] for i in items}), 'items': items}


def collect(state, day, attempt, timestamp):
    for shop in SHOPS:
        accepted = state / 'observations' / day / f'{shop}.json'
        if accepted.exists():
            continue
        try:
            # A separate process bounds network calls and upstream retries.
            result = subprocess.run(['uv', 'run', '--locked', '--project', 'integrations/menu-collector',
                                     'python', '-c', 'import sys; from jacks_menu.download import get_menu_text; print(get_menu_text(sys.argv[1]), end="")', DOCS[shop]], capture_output=True,
                                    text=True, timeout=90, check=True)
            observation = validate_observation(result.stdout, shop, day, timestamp)
        except (subprocess.SubprocessError, UnicodeError) as error:
            observation = {'date': day, 'shop': shop, 'status': 'failed',
                           'fetched_utc': timestamp, 'warnings': [type(error).__name__]}
        immutable(state / 'attempts' / day / f'{attempt}-{shop}.json', observation)
        if observation['status'] == 'valid':
            immutable(accepted, observation)


def trusted_latest():
    manifest = read('artifacts/latest/manifest.json')
    if digest('artifacts/latest/model.joblib') != manifest['outputs']['model.joblib']:
        raise ValueError('Initial model checksum mismatch')
    for path, expected in manifest['processed_data'].items():
        if digest(path) != expected:
            raise ValueError('Frozen history changed')
    for path in ['src/jack_flavours/features.py', 'src/jack_flavours/models.py']:
        if digest(path) != manifest['scientific_inputs'][path]:
            raise ValueError('Frozen scientific implementation changed')
    return joblib.load('artifacts/latest/model.joblib')


def production_model():
    trusted_latest()  # Verify preserved history and the original bootstrap model.
    manifest = read('artifacts/production-v5/manifest.json')
    for path, expected in manifest['implementation_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Production normalization or model implementation changed')
    if digest('artifacts/production-v5/model.joblib') != manifest['model_sha256']:
        raise ValueError('Production v5 model checksum mismatch')
    return joblib.load('artifacts/production-v5/model.joblib')


def bootstrap(state):
    meta = read(f'artifacts/forecasts/latest/forecast-{START}.json')
    csv = Path(f'artifacts/forecasts/latest/forecast-{START}.csv')
    if digest(csv) != meta['csv_sha256']:
        raise ValueError('Bootstrap forecast checksum mismatch')
    if meta['model_sha256'] != digest('artifacts/latest/model.joblib'):
        raise ValueError('Bootstrap forecast model mismatch')
    frame = pd.read_csv(csv)
    for shop, rows in frame.groupby('shop'):
        record = forecast_record(rows, meta, 'adaptive', meta['model_sha256'])
        immutable(state / 'forecasts' / START / f'{shop}-adaptive.json', record)


def forecast_record(rows, metadata, model, model_hash):
    return {'schema': 1, 'issue_date': metadata['issue_date'],
            'target_date': metadata['target_date'], 'created_utc': metadata['created_utc'],
            'shop': str(rows.iloc[0].shop), 'model': model, 'model_sha256': model_hash,
            'train_end': metadata['model_train_end'], 'normalization': metadata.get('normalization', 'original-v1'),
            'history_sha256': metadata.get('history_sha256'),
            'predictions': [{'flavour': str(r.flavour), 'probability': float(r.probability),
                             'rank': int(r.rank), 'top_k': bool(r.top_k)}
                            for r in rows.sort_values('rank').itertuples()]}


def score_forecast(prediction, observation):
    target = datetime.fromisoformat(prediction['target_date']).date()
    issued = datetime.fromisoformat(prediction['created_utc'])
    if issued.tzinfo is None or issued.astimezone(LONDON).date() != target - timedelta(days=1):
        raise ValueError('Forecast was not issued on the preceding London day')
    if prediction['issue_date'] != (target - timedelta(days=1)).isoformat():
        raise ValueError('Incorrect issue date')
    if prediction['train_end'] > prediction['issue_date']:
        raise ValueError('Training cutoff leaks target information')
    if observation['status'] != 'valid' or observation['date'] != str(target):
        raise ValueError('Invalid or mismatched observation')
    if observation['shop'] != prediction['shop']:
        raise ValueError('Shop mismatch')
    items = prediction['predictions']
    labels = [p['flavour'] for p in items]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError('Duplicate or empty prediction labels')
    normalizer = {'v3': canonical_title, 'v4': canonical_title_v4, 'v5': canonical_title_v5}.get(prediction.get('normalization'), lambda x: x)
    actual = {normalizer(n) for n in observation['flavours']}
    y = np.array([name in actual for name in labels], dtype=float)
    p = np.array([item['probability'] for item in items])
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    top = {item['flavour'] for item in items if item['top_k']}
    hits = len(top & actual)
    return {'date': str(target), 'shop': prediction['shop'], 'model': prediction['model'],
            'metric_version': 1, 'normalization': prediction.get('normalization', 'original-v1'), 'hits': hits, 'predicted': len(top), 'actual': len(actual),
            'precision_at_10': hits / len(top) if top else None,
            'recall_at_10': hits / len(actual) if actual else None,
            'brier': float(np.mean((p-y)**2)),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'supported_labels': len(labels), 'supported_actual': int(y.sum()),
            'unsupported_actual': len(actual - set(labels)),
            'observation_sha256': observation['sha256'],
            'model_sha256': prediction['model_sha256']}


def score_pending(state):
    for path in sorted((state / 'forecasts').glob('*/*.json')):
        prediction = read(path)
        observed = state / 'observations' / prediction['target_date'] / f"{prediction['shop']}.json"
        output = state / 'scores' / prediction['target_date'] / path.name
        if output.exists() or not observed.exists():
            continue
        result = score_forecast(prediction, read(observed))
        result.update(forecast_sha256=digest(path), scored_utc=now_utc().isoformat())
        immutable(output, result)
    # Compare only forecasts with the same normalization contract.
    for day in sorted((state / 'observations').glob('*')):
        for shop in SHOPS:
            observation = day / f'{shop}.json'
            if not observation.exists():
                continue
            paths = sorted((state/'forecasts'/day.name).glob(f'{shop}-*.json'))
            for version in ['original-v1', 'v3', 'v4', 'v5']:
                forecasts = [p for p in paths if read(p).get('normalization', 'original-v1') == version]
                output = state/'comparisons'/day.name/f'{shop}-{version}.json'
                if len(forecasts) < 2 or output.exists():
                    continue
                records = [read(p) for p in forecasts]
                for record in records:
                    score_forecast(record, read(observation))
                shared = set.intersection(*[{p['flavour'] for p in r['predictions']} for r in records])
                actual = set(read(observation)['flavours'])
                if version in ('v3', 'v4', 'v5'):
                    normalizer = {'v3': canonical_title, 'v4': canonical_title_v4, 'v5': canonical_title_v5}[version]
                    actual = {normalizer(n) for n in actual}
                scores = {}
                for record in records:
                    probabilities = {p['flavour']: p['probability'] for p in record['predictions']}
                    scores[record['model']] = float(np.mean([(probabilities[n] - (n in actual))**2 for n in sorted(shared)])) if shared else None
                immutable(output, {'date': day.name, 'shop': shop, 'normalization': version,
                                   'shared_labels': len(shared), 'shared_brier': scores,
                                   'forecast_hashes': [digest(p) for p in forecasts]})


def live_dataset(state, day):
    snapshots = pd.read_parquet('data/processed/snapshots.parquet')
    appearances = pd.read_parquet('data/processed/appearances.parquet')
    frozen_end = str(snapshots.snapshot_date.max())
    rows, items = [], []
    for path in sorted((state / 'observations').glob('*/*.json')):
        o = read(path)
        if o['date'] <= frozen_end or o['date'] > day or o['status'] != 'valid':
            continue
        rows.append({'snapshot_date': o['date'], 'shop': o['shop'], 'parse_status': 'valid'})
        items.extend({'snapshot_date': o['date'], 'shop': o['shop'], **item} for item in o['items'])
    # Explicit missing current-day rows retain calendar alignment even with no new observations.
    for shop in SHOPS:
        if day > frozen_end and not any(r['snapshot_date'] == day and r['shop'] == shop for r in rows):
            rows.append({'snapshot_date': day, 'shop': shop, 'parse_status': 'missing'})
    if rows:
        snapshots = pd.concat([snapshots, pd.DataFrame(rows)], ignore_index=True)
    if items:
        appearances = pd.concat([appearances, pd.DataFrame(items)], ignore_index=True)
    appearances = clean_appearances(appearances, confirmed_aliases=True)
    appearances['canonical_name'] = appearances.raw_name.map(canonical_title_v5)
    return Dataset.from_tables(snapshots, appearances, SHOPS)


def issue(state, day, config, frozen, check):
    marker = state / 'issued-v5' / f'{day}.json'
    if marker.exists():
        return read(marker)
    pending = state/'issuance-v5'/f'{day}.json'
    if pending.exists():
        issuance = read(pending)
        model_path = state/'models'/f'v5-{day}.joblib'
        if not model_path.exists() or digest(model_path) != issuance['model_sha256']:
            raise ValueError('Interrupted issuance requires its checksum-verified saved model')
        for record in issuance['records']:
            immutable(state/'forecasts'/record['target_date']/f"{record['shop']}-{record['model']}.json", record)
        result = {'status': 'issued', 'date': day, 'model_sha256': issuance['model_sha256'],
                  'model_asset': f'v5-{day}.joblib', 'history_sha256': issuance['records'][0]['history_sha256'],
                  'shops': sorted({r['shop'] for r in issuance['records']})}
        immutable(marker, result)
        return result
    data = live_dataset(state, day)
    if not data.observed[-1].any():
        return {'status': 'unavailable', 'date': day, 'reason': 'No current-day observations'}
    selection = deepcopy(frozen['selection'])
    selection.update(selected='logit', logistic_C=0.01, min_support=5,
                     test_period={'train_end': day})
    trained = fit_bundle(data, selection, config, check)
    trained.update(recommended='logit', operating_point={'k': 10, 'threshold': .2},
                   training_role='prospective_daily_v5', normalization_version='v5')
    immutable(state/'provenance'/f'v5-{day}.json', {'files': {p: digest(p) for p in ['src/jack_flavours/daily.py', 'src/jack_flavours/cleaning.py', 'src/jack_flavours/live_cleaning.py', 'src/jack_flavours/approved_aliases_v5.json', 'src/jack_flavours/features.py', 'src/jack_flavours/models.py', 'src/jack_flavours/final.py', 'uv.lock', 'integrations/menu-collector/uv.lock']}})
    model_path = state / 'models' / f'v5-{day}.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, model_path)
    model_hash = digest(model_path)
    history_hash = hashlib.sha256(json.dumps(
        {'frozen': read('artifacts/latest/manifest.json')['processed_data'],
         'live': {str(p.relative_to(state)): digest(p) for p in sorted((state/'observations').glob('*/*.json')) if p.parent.name <= day}},
        sort_keys=True).encode()).hexdigest()
    frames = {}
    for name, model, sha in [('adaptive-v5', trained, model_hash), ('frozen-v5', frozen, digest('artifacts/production-v5/model.joblib'))]:
        check()
        frame, meta = build_forecast(model, data, day)
        frames[name] = (frame, meta, sha)
    # Persistence ranks yesterday's full observed menu using prior observed frequencies.
    persistence = []
    for s, shop in enumerate(SHOPS):
        if not data.observed[-1, s]:
            continue
        current = data.presence[-1, s].astype(bool)
        frequency = data.presence[:, s].sum(axis=0)
        order = sorted(range(len(data.names)), key=lambda j: (-int(current[j]), -int(frequency[j]), data.names[j]))
        for rank, j in enumerate(order, 1):
            persistence.append({'shop': shop, 'flavour': data.names[j], 'probability': float(current[j]),
                                'rank': rank, 'top_k': bool(current[j] and rank <= 10)})
    frames['persistence-v5'] = (pd.DataFrame(persistence), frames['adaptive-v5'][1], 'persistence-v5')
    created = now_utc()
    if created.astimezone(LONDON).date().isoformat() != day:
        raise ValueError('Run crossed midnight; refusing retrospective forecast')
    # Construct the entire issuance before writing, and freeze metadata for resumable retries.
    issuance_path = state / 'issuance-v5' / f'{day}.json'
    if issuance_path.exists():
        issuance = read(issuance_path)
    else:
        records = []
        for name, (frame, meta, sha) in frames.items():
            meta.update(created_utc=created.isoformat(), model_train_end=day if name != 'frozen-v5' else frozen['train_end'], history_sha256=history_hash, normalization='v5')
            for shop, rows in frame.groupby('shop'):
                records.append(forecast_record(rows, meta, name, sha))
        issuance = {'records': records, 'model_sha256': model_hash}
        immutable(issuance_path, issuance)
    if issuance['model_sha256'] != model_hash:
        raise ValueError('Refit differs from interrupted issuance; inspect before retrying')
    for record in issuance['records']:
        immutable(state/'forecasts'/record['target_date']/f"{record['shop']}-{record['model']}.json", record)
    result = {'status': 'issued', 'date': day, 'model_sha256': model_hash,
              'model_asset': f'v5-{day}.joblib', 'history_sha256': history_hash,
              'shops': [s for s, valid in zip(SHOPS, data.observed[-1]) if valid]}
    immutable(marker, result)
    return result


def run(state, scheduled=False):
    start = time.monotonic()
    def check():
        if time.monotonic() - start > 1500:
            raise TimeoutError('25-minute pipeline deadline')
    now = now_utc()
    local = now.astimezone(LONDON)
    day = local.date().isoformat()
    if day < START:
        raise ValueError('Before prospective start')
    # Accept delayed same-day starts; never discard an afternoon catch-up.
    if scheduled and local.hour < 11:
        return {'status': 'outside_schedule'}
    state.mkdir(parents=True, exist_ok=True)
    frozen = production_model()
    bootstrap(state)
    attempts = list((state/'runs'/day).glob('*.json'))
    if len(attempts) >= 3 or (state/'issued-v5'/f'{day}.json').exists():
        dashboard(state)
        return {'status': 'already_complete_or_attempt_limit'}
    attempt = len(attempts) + 1
    run_path = state/'runs'/day/f'{attempt}.json'
    immutable(run_path, {'started_utc': now.isoformat(), 'date': day, 'attempt': attempt})
    result = {'date': day, 'attempt': attempt, 'status': 'waiting'}
    try:
        collect(state, day, attempt, now.isoformat())
        score_pending(state)
        complete = all((state/'observations'/day/f'{s}.json').exists() for s in SHOPS)
        if complete or scheduled or (local.hour, local.minute) >= (16, 37) or attempt == 3:
            result.update(issue(state, day, read('configs/experiment.json'), frozen, check))
    except Exception as error:
        result.update(status='failed', error=type(error).__name__, message=str(error))
        raise
    finally:
        immutable(state/'outcomes'/day/f'{attempt}.json', result)
        dashboard(state)
    return result


def dashboard(state):
    scores = [read(p) for p in sorted((state/'scores').glob('*/*.json'))]
    forecasts = [read(p) for p in sorted((state/'forecasts').glob('*/*.json'))]
    observations = [read(p) for p in sorted((state/'observations').glob('*/*.json'))]
    runs = [read(p) for p in sorted((state/'outcomes').glob('*/*.json'))]
    comparisons = [read(p) for p in sorted((state/'comparisons').glob('*/*.json'))]
    output = state/'site'
    output.mkdir(exist_ok=True)
    # Split the full prediction archive by date to stay below Pages' per-file limit.
    archive = output/'forecasts'
    archive.mkdir(exist_ok=True)
    dates = sorted({f['target_date'] for f in forecasts})
    for day in dates:
        (archive/f'{day}.json').write_text(json.dumps([f for f in forecasts if f['target_date'] == day], ensure_ascii=False, allow_nan=False))
    (archive/'index.json').write_text(json.dumps([f'{day}.json' for day in dates]))
    latest = dates[-1] if dates else None
    payload = {'scores': scores, 'forecasts': [f for f in forecasts if f['target_date'] == latest], 'comparisons': comparisons,
               'observations': [{k:v for k,v in o.items() if k not in ('raw','items')} for o in observations],
               'runs': runs, 'generated_utc': now_utc().isoformat()}
    (output/'data.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    template = Path('web/daily.html').read_text()
    (output/'index.html').write_text(template)
    assets = Path('web/assets')
    if assets.is_dir():
        shutil.copytree(assets, output/'assets', dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'dashboard', 'fetch'])
    parser.add_argument('--state', type=Path, default=Path('daily-state'))
    parser.add_argument('--scheduled', action='store_true')
    parser.add_argument('--shop', choices=SHOPS)
    args = parser.parse_args()
    if args.command == 'fetch':
        from jacks_menu.download import get_menu_text
        if not args.shop:
            parser.error('--shop required for fetch')
        print(get_menu_text(DOCS[args.shop]), end='')
    elif args.command == 'dashboard':
        dashboard(args.state)
    else:
        print(json.dumps(run(args.state, args.scheduled), indent=2))


if __name__ == '__main__':
    main()
