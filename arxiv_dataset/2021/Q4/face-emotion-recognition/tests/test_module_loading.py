"""
CI Pytests to check messages on library loading
"""

import os

import pytest

from emotiefflib.facial_analysis import EmotiEffLibRecognizer


def test_unsupported_engine():
    """
    Check exception on unsupported engine
    """
    with pytest.raises(ValueError, match="Unsupported engine specified"):
        _ = EmotiEffLibRecognizer(engine="OpenVINO")


@pytest.mark.skipif(
    os.getenv("WITHOUT_TORCH") is None, reason="Skipping because WITHOUT_TORCH is not set"
)
def test_torch_is_not_installed():
    """
    Check exception when EmotiEffLib recognizer is called for torch without torch installation
    """
    with pytest.raises(ImportError, match="Looks like torch module is not installed: "):
        _ = EmotiEffLibRecognizer(engine="torch")


@pytest.mark.skipif(
    os.getenv("WITHOUT_ONNX") is None, reason="Skipping because WITHOUT_ONNX is not set"
)
def test_onnx_is_not_installed():
    """
    Check exception when EmotiEffLib recognizer is called for ONNX without ONNX installation
    """
    with pytest.raises(ImportError, match="Looks like onnx module is not installed: "):
        _ = EmotiEffLibRecognizer(engine="onnx")
