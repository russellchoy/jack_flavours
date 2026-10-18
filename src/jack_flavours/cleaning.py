"""Versioned descriptive matching; frozen model labels retain their old contract."""
import re
from jack_flavours.data import normalize


def normalize_title(name: str) -> str:
    """Remove conjunction tokens, never substrings such as the 'and' in pandan."""
    return ' '.join(re.sub(r'\band\b|&', ' ', normalize(name)).split())


# User-confirmed pairs, 17 October 2026. No global ingredient sorting.
CONFIRMED_ALIASES = {
    'mascarpone vanilla': 'vanilla mascarpone',
    'brown sugar vanilla': 'vanilla brown sugar',
    'pineapple coconut': 'coconut pineapple',
}


def canonical_title(name: str) -> str:
    normalized = normalize_title(name)
    return CONFIRMED_ALIASES.get(normalized, normalized)


def clean_appearances(appearances, *, confirmed_aliases=True):
    frame = appearances.copy()
    frame['original_canonical_name'] = frame.canonical_name
    frame['canonical_name'] = frame.raw_name.map(canonical_title if confirmed_aliases else normalize_title)
    if frame.canonical_name.eq('').any():
        raise ValueError('Conjunction cleaning produced an empty title')
    # Keep every source row for provenance. Dataset.from_tables builds binary
    # presence, so multiple spellings in one menu count as one appearance.
    return frame
