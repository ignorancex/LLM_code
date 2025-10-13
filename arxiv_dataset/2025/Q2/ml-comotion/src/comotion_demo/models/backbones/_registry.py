# Copyright (C) 2025 Apple Inc. All Rights Reserved.
_lookup = {}


def register_model(fn, model_name=None):
    """Register a model under a name. When unspecified, use the class name."""
    if model_name is None:
        model_name = fn.__name__
    _lookup[model_name] = fn

    return fn
