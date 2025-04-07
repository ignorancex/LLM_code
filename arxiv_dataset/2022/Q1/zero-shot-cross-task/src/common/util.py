import collections
import logging
from pathlib import Path
import re
from unidecode import unidecode
from itertools import groupby

__all__ = [
    "flatten",
    "PROJECT_ROOT",
    "FILE_NAME_CLEANER",
    "DUPE_SPECIAL_CHARS",
    "sanitize_name",
    "all_equal"
]
logger = logging.getLogger(__name__)

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()

FILE_NAME_CLEANER = re.compile(r'[^\w]')
DUPE_SPECIAL_CHARS = re.compile(r'([_\.\-])[_\.\-]+')
LEADING_SPECIAL = re.compile(r'^[^A-Za-z0-9]')
TRAILING_SPECIAL = re.compile(r'[^A-Za-z0-9]+$')


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sanitize_name(name: str) -> str:
    cleaned = FILE_NAME_CLEANER.sub('_', unidecode(name).replace('/', '_')).replace(" ", "_")
    cleaned = DUPE_SPECIAL_CHARS.sub(r'\1', cleaned)
    cleaned = LEADING_SPECIAL.sub('', cleaned)
    return TRAILING_SPECIAL.sub('', cleaned)


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)
