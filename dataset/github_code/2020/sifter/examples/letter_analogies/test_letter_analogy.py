"""Integration test using letter_analogy.py"""
# pylint: disable=pointless-statement,import-error
from external.bazel_python.pytest_helper import main
import letter_analogy

def test_letter_analogy():
    """Regression test for the letter analogy example."""
    solution = letter_analogy.Main(verbose=False)
    assert solution == "efg"

main(__name__, __file__)
