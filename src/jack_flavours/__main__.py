import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Reproducible next-day gelato forecasts')
    parser.add_argument('--version',action='version',version='0.1.0')
    parser.add_argument('--config',type=Path,default=Path('configs/experiment.json'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit',help='Validate pinned raw menus and write snapshot tables')
    p.add_argument('--source',type=Path,default=Path('data/source'))
    p.add_argument('--out',type=Path,default=Path('data/processed'))
    p.add_argument('--report',type=Path,default=Path('reports/data-audit.md'))
    sub.add_parser('freeze',help='Freeze chronological evaluation boundaries without test scoring')
    sub.add_parser('baselines',help='Evaluate temporal baselines on validation only')
    sub.add_parser('validate-models',help='Compare logit and boosting on validation; freeze selection')
    sub.add_parser('evaluate-final',help='One frozen held-out evaluation; returns cached results on repeat')
    p=sub.add_parser('forecast',help='Predict the next snapshot from the saved model, without refitting')
    p.add_argument('--issue-date',help='ISO issue date; defaults to latest date in pinned history')
    p.add_argument('--out',type=Path)
    p.add_argument('--model',choices=['latest','evaluation'],help='Default: latest refit when available, otherwise evaluation model')
    sub.add_parser('refit-latest',help='Refit using all audited history and compare next-day forecasts')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.command=='refit-latest':
        from .refit import refit_latest
        print(json.dumps(refit_latest(config),indent=2))
    if args.command=='forecast':
        from .forecast import forecast
        print(json.dumps(forecast(config,args.issue_date,args.out,Path('artifacts/latest' if args.model=='latest' else 'artifacts/final') if args.model else None),indent=2))
    if args.command=='evaluate-final':
        from .final import evaluate_final
        print(json.dumps(evaluate_final(config),indent=2))
    if args.command=='validate-models':
        from .experiment import validate_models
        print(json.dumps(validate_models(config),indent=2))
    if args.command=='baselines':
        from .experiment import baselines
        print(json.dumps(baselines(config),indent=2))
    if args.command=='freeze':
        from .features import Dataset, freeze
        print(json.dumps(freeze(Dataset.load(Path('data/processed'),config['shops']),config,Path('configs/splits.json'),Path('reports/evaluation-protocol.md')),indent=2))
    if args.command=='audit':
        from .data import audit
        print(json.dumps(audit(args.source,args.out,args.report,config),indent=2))

if __name__=='__main__':
    main()
