# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Backbone network for CoMotion."""

from ._registry import _lookup
from .convnext import ConvNextV2  # noqa


def initialize(backbone_choice):
    """Initialize a registered backbone."""
    assert backbone_choice in _lookup, (
        f"Backbone choice '{backbone_choice}' not found in registry."
    )
    return _lookup[backbone_choice]()
