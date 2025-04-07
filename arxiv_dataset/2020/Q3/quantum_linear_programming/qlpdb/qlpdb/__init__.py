"""Prepares the usage of the qlpdb module
"""
import os
import sys
from django import setup as _setup


def _init():
    """Initializes the django environment for qlpdb
    """
    if sys.version_info.major < 3:
        raise RuntimeError(
            "You must run this code with Python major version 3"
            " because we use hashing to store data."
        )

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qlpdb.config.settings")
    _setup()


_init()
