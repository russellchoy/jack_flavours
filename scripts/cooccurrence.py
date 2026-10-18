"""Within-shop flavour co-occurrence on valid menus; descriptive, not causal."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cooccurrence(data):
    rows=[]
    for si,shop in enumerate(data.shops):
        # Wide integer type avoids overflow when common pairs exceed 255 days.
        x=data.presence[data.observed[:,si],si,:].astype(np.int64)
        n=len(x)
        if not n: continue
        totals=x.sum(axis=0); joint=x.T@x
        for i,j in zip(*np.triu_indices(len(data.names),k=1)):
            ab=int(joint[i,j])
            if not ab: continue
            a,b=int(totals[i]),int(totals[j])
            rows.append(dict(shop=shop,flavour_a=data.names[i],flavour_b=data.names[j],valid_menus=n,
                menus_a=a,menus_b=b,shared_menus=ab,joint_pct=100*ab/n,
                b_given_a_pct=100*ab/a,a_given_b_pct=100*ab/b,
                expected_shared=a*b/n,lift=ab*n/(a*b),
                eligible_association=a>=20 and b>=20 and ab>=10))
    return pd.DataFrame(rows,columns=['shop','flavour_a','flavour_b','valid_menus','menus_a','menus_b','shared_menus','joint_pct','b_given_a_pct','a_given_b_pct','expected_shared','lift','eligible_association'])


def cooccurrence_section(data,out,shops):
    pairs=cooccurrence(data)
    pairs.to_csv(out/'flavour-cooccurrence.csv',index=False)
    fig,axes=plt.subplots(len(shops),2,figsize=(20,14),layout='constrained',squeeze=False)
    tables=[]
    for si,(shop,label) in enumerate(shops.items()):
        group=pairs[pairs.shop.eq(shop)]
        popular=group.sort_values(['shared_menus','flavour_a','flavour_b'],ascending=[False,True,True]).head(12)
        associated=group[group.eligible_association].sort_values(['lift','shared_menus','flavour_a','flavour_b'],ascending=[False,False,True,True]).head(12)
        for ax,part,key,title in [(axes[si,0],popular,'shared_menus','Most frequent pairs'),(axes[si,1],associated,'lift','Strongest relative associations')]:
            part=part.iloc[::-1]
            names=[textwrap.fill(f'{a} + {b}',43) for a,b in zip(part.flavour_a,part.flavour_b)]
            ax.barh(np.arange(len(part)),part[key],color='#176c80' if key=='shared_menus' else '#d88935')
            ax.set_yticks(np.arange(len(part)),names,fontsize=8)
            ax.set_title(f'{label} · {title}',loc='left')
            ax.set_xlabel('Valid menus containing both flavours' if key=='shared_menus' else 'Lift: observed / independence-expected co-occurrence')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='x',alpha=.2)
            if key=='lift':ax.axvline(1,color='#666',linestyle='--',linewidth=1)
            if part.empty:ax.text(.5,.5,'No pairs meet the support rule',transform=ax.transAxes,ha='center')
        for title,part in [('Most frequent pairs',popular),('Strongest relative associations',associated)]:
            display=part[['flavour_a','flavour_b','menus_a','menus_b','shared_menus','b_given_a_pct','a_given_b_pct','lift']].rename(columns={'flavour_a':'Flavour A','flavour_b':'Flavour B','menus_a':'A menus','menus_b':'B menus','shared_menus':'Together','b_given_a_pct':'B given A (%)','a_given_b_pct':'A given B (%)','lift':'Lift'})
            tables.append(f'<h3>{label}: {title.lower()}</h3><div class="wide">'+display.to_html(index=False,border=0,float_format=lambda v:f'{v:.2f}')+'</div>')
    fig.suptitle('Which flavours appear alongside each other?',fontsize=20)
    fig.savefig(out/'flavour-cooccurrence.png',dpi=150);plt.close(fig)
    return '''<h2 id="cooccurrence">Which flavours appear alongside each other?</h2>
<p>Each observation is one valid menu at one shop on one date. Flavours are complete menu items, not ingredients inside a recipe. Missing menus are excluded. The left panels show the 12 pairs seen together most often; the right panels adjust for how common each flavour is.</p>
<img src="flavour-cooccurrence.png" alt="Most frequent flavour pairs and strongest relative associations by shop">
<p><b>Conditional probability:</b> “B given A” is the percentage of menus containing A that also contain B. Its reverse can differ. <b>Lift</b> = P(A and B) ÷ [P(A) × P(B)]. A lift of 2 means twice the co-occurrence expected if their appearances were independent within that shop; 1 means that independence benchmark. It is not a twofold probability of tomorrow’s appearance.</p>
<p>Relative-association rankings require each flavour on at least <b>20 valid menus</b> and the pair together on at least <b>10</b>. These are descriptive support filters, not significance tests. Consecutive menu days are dependent, and seasonal availability or long shared runs can drive high lift; the calculation does not adjust for these. It does not establish intentional pairing or customer preferences.</p>
<details><summary>Pair counts and conditional probabilities</summary>'''+''.join(tables)+'''</details><p><a href="flavour-cooccurrence.csv">Download all observed pair counts, conditional probabilities and lift</a>. Unobserved pairs have zero co-occurrences and are omitted from this file; self-pairs are excluded. All names use the latest confirmed cleaning.</p>'''
