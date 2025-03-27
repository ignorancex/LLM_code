"""Tests for utils.py"""
from external.bazel_python.pytest_helper import main
import runtime.utils as utils

def test_freeze_thaw_dict():
    """Tests freezedict/thawdict."""
    x = dict({"hello": "there"})
    assert utils.thawdict(utils.freezedict(x)) == x
    assert isinstance(utils.freezedict(x), tuple)

def test_is_empty():
    """Tests is_empty(...)."""
    assert utils.is_empty(x for x in range(0))
    assert not utils.is_empty(x for x in range(5))

def test_translator():
    """Tests the Translator class."""
    translator = utils.Translator(dict({
        "hello": "bonjour",
        "why": "porquoi",
        "what": "quoi",
    }))
    assert translator.translate("hello") == "bonjour"
    assert translator.translate("Matthew") == "Matthew"

    assert (translator.translate_tuple(("hello", "Matthew"))
            == ("bonjour", "Matthew"))

    assert (translator.translate_tuples([("hello", "Matthew"),
                                         ("why", "what")])
            == [("bonjour", "Matthew"), ("porquoi", "quoi")])

    assert (translator.translate_list(["hello", "Matthew"])
            == ["bonjour", "Matthew"])

    composed = translator.compose(dict({
        "bonjour": "salam",
        "porquoi": "chera",
        "merci": "merci",
    }))
    assert composed == dict({"hello": "salam", "why": "chera"})
    composed = translator.compose(dict({
        "bonjour": "salam",
        "porquoi": "chera",
        "merci": "merci",
    }), default_identity=True)
    assert composed == dict({"hello": "salam", "why": "chera", "what": "quoi"})

    concatenated = translator.concatenated_with(dict({
        "thanks": "merci",
    }))
    assert concatenated == dict({
        "hello": "bonjour",
        "why": "porquoi",
        "what": "quoi",
        "thanks": "merci",
    })

def test_real_hash():
    """Regression test for the real_hash method."""
    truth = "ff42aa4718d9da1c7d491e8be8116f0c62db8d910de16e8ee0648147"
    assert utils.real_hash(dict({"hello": "there"})) == truth
    try:
        # We don't yet support tuples, it should throw a NIE.
        utils.real_hash(("hi", "hello"))
    except NotImplementedError:
        pass
    else:
        assert False

main(__name__, __file__)
