"""Descriptive flavour charts; never modifies model artifacts."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/jack-flavours-matplotlib')
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jack_flavours.features import Dataset
from jack_flavours.cleaning import clean_appearances
from menu_durations import spells
from cooccurrence import cooccurrence_section

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/flavour-statistics'
SHOPS = {'all_saints': 'All Saints', 'benet_street': 'Bene’t Street'}


def summarise(data):
    rows, events = [], []
    for si, shop in enumerate(data.shops):
        valid = data.observed[:, si]
        for fi, name in enumerate(data.names):
            state = data.presence[:, si, fi].astype(np.int8)
            state[~valid] = -1
            seen = np.flatnonzero(state == 1)
            if not len(seen):
                continue
            ss = spells(state)
            stays = [x['days'] for x in ss if x['state'] == 1 and x['completed']]
            gaps = [x['days'] for x in ss if x['state'] == 0 and x['completed']]
            recorded = []
            for a, b in zip(seen[:-1], seen[1:]):
                if b-a > 1 and state[a+1] == 0:
                    recorded.append(int(b-a-1))
                    events.append(dict(shop=shop, flavour=name, kind='recorded_return', days=int(b-a-1), start=str(data.dates[a+1].date()), complete=bool(valid[a+1:b].all())))
            for x in ss:
                events.append(dict(shop=shop, flavour=name, kind='stay' if x['state'] else 'absence', days=x['days'], start=str(data.dates[x['start']].date()), complete=x['completed']))
            row = dict(shop=shop, flavour=name, appearances=len(seen), valid_menus=int(valid.sum()), menu_share_pct=100*len(seen)/valid.sum(), first_seen=str(data.dates[seen[0]].date()), last_seen=str(data.dates[seen[-1]].date()), confirmed_departures=sum(x['state']==0 for x in ss), recorded_returns=len(recorded))
            for key, vals in [('stay', stays), ('absence', gaps), ('recorded_return', recorded), ('between_snapshots', np.diff(seen))]:
                row[key+'_n'] = len(vals)
                row[key+'_mean'] = float(np.mean(vals)) if len(vals) else np.nan
                row[key+'_median'] = float(np.median(vals)) if len(vals) else np.nan
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(events)


def chart(df, key, filename, title, all_flavours=False):
    fig, axes = plt.subplots(2, 1, figsize=(max(15, df.groupby('shop').size().max()*.105) if all_flavours else 16, 13), layout='constrained')
    for ax, (shop, label) in zip(axes, SHOPS.items()):
        part = df[df.shop == shop].copy()
        if not all_flavours:
            part = part.nlargest(30, 'appearances')
        part = part.sort_values([key+'_median', 'flavour'], na_position='last')
        x = np.arange(len(part))
        ax.scatter(x, part[key+'_mean'], s=18 if all_flavours else 40, c='#e68a2e', label='Mean', zorder=3)
        ax.scatter(x, part[key+'_median'], s=12 if all_flavours else 28, c='#176c80', label='Median', zorder=4)
        # Missing duration estimates stay blank, never become zero-day observations.
        ax.set_xticks(x, part.flavour, rotation=90, fontsize=4 if all_flavours else 8)
        ax.set_ylabel('Calendar days'); ax.set_title(label + f' · {len(part)} flavours', loc='left')
        ax.grid(axis='y', alpha=.2); ax.set_ylim(bottom=0); ax.legend(loc='upper center')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontsize=18)
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def menu_similarity(data):
    """Daily Jaccard similarity; unknown or empty-union days remain undefined."""
    if len(data.shops) != 2:
        raise ValueError('Menu similarity requires exactly two shops')
    both = data.observed.all(axis=1)
    present = data.presence.astype(bool)
    shared = (present[:, 0] & present[:, 1]).sum(axis=1)
    union = (present[:, 0] | present[:, 1]).sum(axis=1)
    score = np.divide(100.0*shared, union, out=np.full(len(union), np.nan), where=both & (union>0))
    return pd.DataFrame({'date': data.dates, 'both_menus_valid': both,
                         'shared_flavours': np.where(both, shared, np.nan),
                         'distinct_flavours': np.where(both, union, np.nan),
                         'similarity_pct': score})


def similarity_chart(data):
    frame = menu_similarity(data)
    frame.to_csv(OUT/'daily-menu-similarity.csv', index=False)
    fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
    ax.plot(frame.date, frame.similarity_pct, color='#176c80', linewidth=.9,
            marker='.', markersize=2.5, label='Daily menu similarity')
    mean = frame.similarity_pct.mean()
    ax.axhline(mean, color='#e68a2e', linestyle='--', label=f'Mean across valid paired dates: {mean:.1f}%')
    ax.set_ylim(0,100)
    ax.set_ylabel('Shared / distinct flavours (%)'); ax.set_xlabel('Date')
    ax.set_title('How similar are the two shop menus? · All Saints vs Bene’t Street', loc='left', fontsize=16)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper left')
    fig.savefig(OUT/'daily-menu-similarity.png',dpi=170); plt.close(fig)
    return frame



def menu_autocorrelogram(data, max_lag=180):
    """Mean within-shop Jaccard by exact calendar-day lag, valid endpoints only."""
    if max_lag < 0:
        raise ValueError('max_lag must be nonnegative')
    if not data.dates.equals(pd.date_range(data.dates[0], periods=len(data.dates))):
        raise ValueError('Expected a complete daily calendar, including missing menus')
    rows = []
    for si, shop in enumerate(data.shops):
        presence = data.presence[:, si].astype(bool)
        for lag in range(min(max_lag, len(data.dates)-1)+1):
            stop = len(data.dates)-lag
            left, right = presence[:stop], presence[lag:]
            union = (left | right).sum(axis=1)
            valid = data.observed[:stop, si] & data.observed[lag:, si] & (union > 0)
            scores = 100.0 * (left & right).sum(axis=1)[valid] / union[valid]
            rows.append(dict(shop=shop, lag_days=lag, pairs=len(scores),
                             similarity_pct=float(scores.mean()) if len(scores) else np.nan))
    return pd.DataFrame(rows)


def autocorrelogram_chart(data):
    frame = menu_autocorrelogram(data)
    frame.to_csv(OUT/'menu-autocorrelogram.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
    for shop, label in SHOPS.items():
        part = frame[frame.shop == shop]
        axes[0].plot(part.lag_days, part.similarity_pct, label=label, linewidth=1.6)
        axes[1].plot(part.lag_days, part.pairs, label=label)
    axes[0].set(title='How quickly do menus change? · within-shop similarity',
                ylabel='Mean Jaccard similarity (%)', ylim=(0, 103))
    axes[1].set(xlabel='Calendar-day lag x: menu at t compared with t + x',
                ylabel='Number of menu comparisons', ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend()
    fig.savefig(OUT/'menu-autocorrelogram.png', dpi=170)
    plt.close(fig)
    return frame

def main():
    os.chdir(ROOT); OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path('configs/experiment.json').read_text())
    snapshots = pd.read_parquet('data/processed/snapshots.parquet')
    appearances = clean_appearances(pd.read_parquet('data/processed/appearances.parquet'))
    cleaned = Path('data/cleaned-descriptive-v3')
    cleaned.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(cleaned/'snapshots.parquet', index=False)
    appearances.to_parquet(cleaned/'appearances.parquet', index=False)
    aliases = appearances.groupby(['original_canonical_name', 'canonical_name']).size().rename('source_rows').reset_index()
    aliases.to_csv(OUT/'title-mapping.csv', index=False)
    data = Dataset.from_tables(snapshots, appearances, config['shops'])
    cooccurrence_html = cooccurrence_section(data, OUT, SHOPS)
    autocorrelogram_chart(data)
    similarity = similarity_chart(data)
    df, events = summarise(data)
    assert (df.appearances <= df.valid_menus).all()
    assert df.appearances.sum() == data.presence.sum()
    assert (events.days >= 1).all()
    assert (df.recorded_returns <= df.confirmed_departures).all()
    df.to_csv(OUT/'flavour-statistics.csv', index=False)
    events.to_csv(OUT/'duration-events.csv', index=False)
    common = df.groupby('flavour').appearances.sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 9), layout='constrained')
    top = common.head(25).sort_values()
    ax.barh(top.index, top.values, color='#176c80')
    ax.set_xlabel('Recorded appearances (one flavour at one shop on one valid date)')
    ax.set_title('The menu regulars · top 25 across both shops', loc='left', fontsize=17)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'most-common.png', dpi=160); plt.close(fig)
    for key, title in [('stay','How long do flavours stay? · completed uninterrupted menu stays'), ('absence','How long until they return? · fully observed days off-menu'), ('recorded_return','The longer view · days off until the next recorded return')]:
        chart(df,key,key+'-common.png',title)
        chart(df,key,key+'-all.png',title,True)
    # A distribution view exposes variation hidden by averages.
    fig, axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax, key, title in zip(axes,['stay','absence'],['Completed menu stays','Completed uninterrupted absences']):
        for shop,label in SHOPS.items():
            vals = np.sort(events.loc[(events.shop==shop)&(events.kind==key)&events.complete,'days'].to_numpy())
            ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=f'{label} (n={len(vals):,})')
        ax.set_xscale('log');ax.set_xlabel('Days (log scale)');ax.set_ylabel('Share lasting this long or less');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'duration-distributions.png',dpi=160);plt.close(fig)
    pooled=common.head(15).rename('Recorded appearances').reset_index()
    pooled['Share of valid shop menus (%)']=(100*pooled['Recorded appearances']/data.observed.sum()).round(1)
    def table(part):
        return part.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:.1f}')
    sections=[]
    for shop,label in SHOPS.items():
        part=df[df.shop==shop].nlargest(20,'appearances')[['flavour','appearances','menu_share_pct','stay_n','stay_mean','stay_median','absence_n','absence_mean','absence_median','recorded_return_n','recorded_return_mean','recorded_return_median']]
        part=part.rename(columns={'flavour':'Flavour','appearances':'Menus','menu_share_pct':'Menu %','stay_n':'Stays n','stay_mean':'Stay mean','stay_median':'Stay median','absence_n':'Strict returns n','absence_mean':'Strict gap mean','absence_median':'Strict gap median','recorded_return_n':'Recorded returns n','recorded_return_mean':'Recorded gap mean','recorded_return_median':'Recorded gap median'})
        sections.append(f'<h2>{label}: the 20 most common</h2><div class="wide">{table(part)}</div>')
    plots=''
    for key,title,desc in [('stay','Time on the menu','Completed stays only: both arrival and departure must be observed, without any missing days.'),('absence','Time off the menu','Completed absences with every intervening day observed. This favours shorter gaps; flavours with no observed return are excluded.'),('recorded_return','Longer gaps between recorded returns','After a confirmed departure, elapsed off-menu days until the next recorded appearance. Missing menus may hide an earlier return, so this is not proven continuous absence.')]:
        plots+=f'<h2>{title}</h2><p>{desc} Each panel shows its shop’s 30 most common flavours, sorted by median duration; blank columns mean no qualifying events. Orange = mean; teal = median. Counts are in the tables and CSV.</p><img src="{key}-common.png"><details><summary>Explore every flavour (scroll horizontally)</summary><p>Includes rare flavours with very little evidence. An estimate based on one event is not a reliable typical duration.</p><div class="wide"><img class="all" src="{key}-all.png"></div><a href="{key}-all.png">Open full-resolution chart</a></details>'
    report=f'''<!doctype html><html><head><meta charset="utf-8"><title>Jack’s flavour field notes</title><style>body{{font:16px/1.6 system-ui;color:#20333c;background:#faf8f3;max-width:1200px;margin:50px auto;padding:0 24px}}h1{{font-size:44px}}h2{{margin-top:48px}}img{{max-width:100%;background:white;border-radius:8px}}.wide{{overflow-x:auto}}img.all{{max-width:none;width:auto;height:1000px}}table{{border-collapse:collapse;background:white;font-size:13px;white-space:nowrap}}td,th{{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}}summary{{cursor:pointer;color:#176c80;font-weight:bold}}a{{color:#176c80}}.note{{padding:20px;background:#e9f0eb;border-radius:10px}}</style></head><body>
    <h1>Jack’s flavour field notes</h1><p>Descriptive menu history · {data.dates[0].date()} to {data.dates[-1].date()}</p>
    <div class="note"><b>{int(data.observed.sum()):,} valid shop menus · {len(common):,} distinct recorded flavour labels · {int(common.sum()):,} flavour appearances.</b><br>These describe the recorded menus, not sales or predictions. Matching removes standalone “and” and ampersands from all titles and collapses whitespace. Three user-confirmed ingredient-order aliases are also merged: vanilla mascarpone, vanilla brown sugar and coconut pineapple. Original names are preserved; other spelling variants can remain separate.</div>
    <h2>How similar is a menu to itself later?</h2><p>For each shop and calendar-day lag x, compare the sets of flavours at t and t + x using Jaccard similarity, then average equally over valid date pairs. Lag zero is 100%. Missing endpoints are excluded; intervening missing days do not change the calendar lag. The lower panel counts the menu comparisons used in the average above. For example, at a seven-day lag, Monday’s menu compared with the following Monday’s menu is one comparison. Both menus must be recorded and parsed successfully; otherwise that comparison is excluded.</p><img src="menu-autocorrelogram.png"><p>This is a menu-similarity correlogram, not a Pearson autocorrelation or fitted AR(1). Repeated favourites can produce a positive long-run floor, and seasonal or weekly recurrence can produce bumps. No declining shape is imposed. Overlapping pairs are dependent; no independent-pair confidence intervals are shown. <a href="menu-autocorrelogram.csv">Download lag statistics</a> · <a href="title-mapping.csv">Inspect title matching</a>.</p>
    <h2>Do the shops serve the same flavours?</h2><p>Daily similarity = shared flavours ÷ distinct flavours across both menus × 100 (Jaccard similarity). Identical nonempty menus score 100%; menus with no overlap score 0%. Dates missing either valid menu are gaps, not zeros. Conjunction-normalized flavour labels are compared; other unmerged naming variants can reduce measured overlap.</p><img src="daily-menu-similarity.png"><p>Across {int(similarity.similarity_pct.notna().sum()):,} comparable dates, mean similarity is <b>{similarity.similarity_pct.mean():.1f}%</b> and median is <b>{similarity.similarity_pct.median():.1f}%</b>. Each date has equal weight. <a href="daily-menu-similarity.csv">Download daily counts and percentages</a>.</p>
    <p>The 100% spike on 7 October 2025 reflects identical sets of 14 canonical labels in the two recorded menus. It is retained as recorded; these data alone cannot distinguish a genuinely identical menu from a collection duplication.</p><h2>The regulars</h2><img src="most-common.png">{table(pooled)}
    {cooccurrence_html}
    {plots}<h2>A view of the whole distribution</h2><p>The curves pool completed events within each shop; frequently recurring flavours contribute more events. For example, a curve at 0.8 means 80% of included events lasted that many days or less.</p><img src="duration-distributions.png">
    {''.join(sections)}
    <h2>Reading the gaps correctly</h2><p>Present on Monday, absent Tuesday and Wednesday, back Thursday = two days off-menu. Consecutive daily appearances are not separate returns. CSV columns <code>between_snapshots_mean/median</code> additionally give elapsed days between every consecutive recorded appearance, including one-day intervals within a stay; these answer a different question.</p><p>Missing/failed menus are unknown, never absence. Stays crossing a missing day or ending at the history boundary are censored and excluded from completed-stay means. Strict absence estimates have the same limitation. Recorded-return estimates allow intervening unknown days but exclude departures with no later recorded return. None of these conditional means estimates an unconditional expected wait, and rare flavours have noisy averages. First recorded appearance is not necessarily a recipe’s launch date.</p>
    <h2>Download and reproduce</h2><p><a href="flavour-statistics.csv">All per-flavour/shop statistics</a> · <a href="duration-events.csv">Underlying duration events</a></p><p>Run <code>.venv/bin/python scripts/flavour_descriptives.py</code> from the project. Source commit: <code>{html.escape(config['source_commit'])}</code>. Mean and median are calculated per flavour/shop; combined popularity counts each shop/date separately. Model artifacts are unchanged.</p></body></html>'''
    (OUT/'index.html').write_text(report)
    print(json.dumps({'report':str(OUT/'index.html'),'flavours':len(common),'valid_menus':int(data.observed.sum()),'top5':common.head().to_dict(),'shop_flavour_rows':len(df),'events':len(events)},indent=2))


if __name__=='__main__':
    main()
