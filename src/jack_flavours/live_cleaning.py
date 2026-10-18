"""Approved live aliases added after the frozen v3 comparison."""
from .cleaning import canonical_title

# Explicit user approval on 18 October 2026; never global ingredient sorting.
LIVE_ALIASES = {'house vanilla': 'vanilla'}


def canonical_title_v4(name):
    cleaned = canonical_title(name)
    return LIVE_ALIASES.get(cleaned, cleaned)


# Every mapping below comes from the user-approved 18 October candidate review.
# Keep v3/v4 functions unchanged so their archived forecasts can still be scored.
import json
from pathlib import Path

APPROVED_ALIASES_V5 = json.loads(Path(__file__).with_name('approved_aliases_v5.json').read_text())['aliases']


def canonical_title_v5(name):
    cleaned = canonical_title_v4(name)
    return APPROVED_ALIASES_V5.get(cleaned, cleaned)
