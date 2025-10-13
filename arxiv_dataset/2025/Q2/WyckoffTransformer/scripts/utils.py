import gzip
import json
from io import TextIOWrapper
from pathlib import Path


def dump_json_to_gz(file_name: Path, data):
    """
    Writes an object to a gzipped JSON file without the timestamp.
    """
    if file_name.suffixes != ['.json', '.gz']:
        raise ValueError('File name must have .json.gz extension')
    with gzip.GzipFile(file_name, 'w', mtime=0) as f:
        with TextIOWrapper(f, encoding='ascii') as t:
            json.dump(data, t)