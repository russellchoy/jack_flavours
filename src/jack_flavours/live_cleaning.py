"""Approved live aliases added after the frozen v3 comparison."""
from .cleaning import canonical_title

# Explicit user approval on 18 October 2026; never global ingredient sorting.
LIVE_ALIASES = {'house vanilla': 'vanilla'}


def canonical_title_v4(name):
    cleaned = canonical_title(name)
    return LIVE_ALIASES.get(cleaned, cleaned)
