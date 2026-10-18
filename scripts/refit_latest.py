"""Manually refit/reuse the latest audited model and compare tomorrow forecasts."""
from pathlib import Path
import os
import json
from jack_flavours.refit import refit_latest

if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parents[1])
    print(json.dumps(refit_latest(json.loads(Path('configs/experiment.json').read_text())),indent=2))
