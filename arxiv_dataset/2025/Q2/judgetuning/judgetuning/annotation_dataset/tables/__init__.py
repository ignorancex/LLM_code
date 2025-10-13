from pathlib import Path


def default_table_path():
    return Path("~/judge-tuning-data/tables/").expanduser()
