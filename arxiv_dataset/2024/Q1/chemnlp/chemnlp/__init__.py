"""Version number."""
__version__ = "2023.8.1"

import os


def test(*args):
    """Run pytest in the base of chemnlp."""
    import pytest

    path = os.path.join(os.path.split(__file__)[0], "tests")
    pytest.main(args=[path] + list(args))
