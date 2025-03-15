"""Initialize the modules module."""

from truetype_vs_postscript_transformer.models.classifier.cls_token import (
    ClsTokenFontClassifier,
)
from truetype_vs_postscript_transformer.models.classifier.t3 import T3

__all__ = [
    "T3",
    "ClsTokenFontClassifier",
]
