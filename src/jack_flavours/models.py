"""Small regularized models with safe single-class fallbacks."""
from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

@dataclass
class ConstantProbability:
    probability: float
    def predict_proba(self,x):
        p=np.full(len(x),self.probability)
        return np.column_stack([1-p,p])


def fit_binary(x,y,kind,config,C=1.0):
    classes=np.unique(y)
    if len(classes)==1: return ConstantProbability(float(classes[0]))
    if kind=='logit':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='lbfgs',max_iter=1000,random_state=config['seed']))
    elif kind=='boosting':
        model=HistGradientBoostingClassifier(random_state=config['seed'],**config['boosting'])
    else:raise ValueError(f'Unknown model: {kind}')
    return model.fit(x,y)


def choose_model(summary):
    base=['frequency','persistence']
    def clears(name):
        m=summary[name]['pooled']
        return all(m['micro_ap']>summary[b]['pooled']['micro_ap'] and m['brier']<=summary[b]['pooled']['brier'] for b in base)
    logit_ok=clears('logit'); boost_ok=clears('boosting')
    l=summary['logit']; b=summary['boosting']
    folds_better=sum(bf['metrics']['micro_ap']>lf['metrics']['micro_ap'] for bf,lf in zip(b['folds'],l['folds']))
    # Weight shop AP by its forecast count across the three folds for a consistency check.
    shops=sorted(l['folds'][0]['by_shop'])
    def shop_ap(m,shop):
        weights=[f['by_shop'][shop]['forecasts'] for f in m['folds']]
        return sum(f['by_shop'][shop]['micro_ap']*w for f,w in zip(m['folds'],weights))/sum(weights)
    both_shops=all(shop_ap(b,s)>shop_ap(l,s) for s in shops)
    substantial=b['pooled']['micro_ap']>=l['pooled']['micro_ap']+0.01 and b['pooled']['brier']<=l['pooled']['brier'] and folds_better>=2 and both_shops
    baseline=max(base,key=lambda m:(summary[m]['pooled']['micro_ap'],-summary[m]['pooled']['brier']))
    selected='boosting' if boost_ok and (not logit_ok or substantial) else ('logit' if logit_ok else baseline)
    return {'selected':selected,'baseline_fallback':baseline,'logit_passes_validation_gate':logit_ok,'boosting_passes_validation_gate':boost_ok,'boosting_substantially_beats_logit':substantial,'boosting_better_folds':folds_better,'boosting_better_both_shops':both_shops,'final_test_status':'not_evaluated'}
