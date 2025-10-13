#!/usr/bin/env python3

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gcgmics.settings import dict_to_namespace, find_placeholders_with_path, get


def test_find_placeholders_with_path_simple():
    data = {"key": "value {placeholder}"}
    result = find_placeholders_with_path(data)
    assert result == {"key": ["placeholder"]}


def test_find_placeholders_with_path_nested_dict():
    data = {"parent": {"child": "value {nested_placeholder}"}}
    result = find_placeholders_with_path(data)
    assert result == {"parent.child": ["nested_placeholder"]}


def test_find_placeholders_with_path_list():
    data = ["list item {list_placeholder}", {"key": "value {dict_placeholder}"}]
    result = find_placeholders_with_path(data)
    assert result == {"[0]": ["list_placeholder"], "[1].key": ["dict_placeholder"]}


def test_find_placeholders_with_path_complex():
    data = {"a": "text {var1}", "b": {"c": ["item {var2}", {"d": "text {var3}"}]}}
    result = find_placeholders_with_path(data)
    assert result == {"a": ["var1"], "b.c[0]": ["var2"], "b.c[1].d": ["var3"]}


def test_dict_to_namespace_flat():
    data = {"key1": "value1", "key2": 42}
    result = dict_to_namespace(data)
    assert isinstance(result, SimpleNamespace)
    assert result.key1 == "value1"
    assert result.key2 == 42


def test_dict_to_namespace_nested():
    data = {"parent": {"child": "nested_value", "numbers": [1, 2, 3]}}
    result = dict_to_namespace(data)
    assert isinstance(result, SimpleNamespace)
    assert isinstance(result.parent, dict)  # Should remain a dict
    assert result.parent["child"] == "nested_value"
    assert result.parent["numbers"] == [1, 2, 3]


def test_dict_to_namespace_with_lists():
    data = {"list": [{"item1": "value1"}, {"item2": "value2"}]}
    result = dict_to_namespace(data)
    assert isinstance(result, SimpleNamespace)
    assert isinstance(result.list, list)
    assert isinstance(result.list[0], dict)  # Should remain a dict
    assert result.list[0]["item1"] == "value1"
    assert result.list[1]["item2"] == "value2"


class MockSettings:
    def __init__(self):
        self.top = "top_value"
        self.nested = SimpleNamespace(
            value=42, lst=[SimpleNamespace(a=1), SimpleNamespace(b=2)]
        )


def test_get_top_level(monkeypatch):
    monkeypatch.setattr("gcgmics.settings.load_settings", lambda: MockSettings())
    assert get("top") == "top_value"


def test_get_nested_value():
    settings = MockSettings()
    with patch("gcgmics.settings.load_settings", return_value=settings):
        assert get("nested.value") == 42


def test_get_list_index():
    settings = MockSettings()
    with patch("gcgmics.settings.load_settings", return_value=settings):
        assert get("nested.lst[0].a") == 1
        assert get("nested.lst[1].b") == 2


def test_get_missing_path_default():
    settings = MockSettings()
    with patch("gcgmics.settings.load_settings", return_value=settings):
        assert get("invalid.path", "default") == "default"
        assert get("nested.lst[2].c", 99) == 99


def test_get_invalid_index():
    settings = MockSettings()
    with patch("gcgmics.settings.load_settings", return_value=settings):
        assert get("nested.lst[invalid].a", "error") == "error"
