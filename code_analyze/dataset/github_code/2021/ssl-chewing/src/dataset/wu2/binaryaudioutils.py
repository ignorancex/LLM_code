import struct
from pathlib import Path
from typing import IO

import numpy as np

_INDIANNESS: str = "<"
"""Based on 'ieee-le' of MATLAB  (see ``src/main/matlab/dataset/wu2/generate_audio_files.m``)"""

_SAMPLE_TYPE: str = "f"
"""Based on 'single' of MATLAB (see ``src/main/matlab/dataset/wu2/generate_audio_files.m``)"""

_SAMPLE_BYTES: int = 4
"""Based on ``_SAMPLE_TYPE``"""


def get_file_length(file: Path) -> int:
    """Returns how many audio samples are contained in the binary audio file ``file``."""
    assert isinstance(file, Path) and file.exists()

    b: int = file.stat().st_size
    if b % _SAMPLE_BYTES != 0:
        raise ValueError("File size (" + str(b) + "B) is not a valid length")

    return int(b / _SAMPLE_BYTES)


def _decode_bytes(b: bytes) -> np.ndarray:
    """Decodes a byte stream of audio samples to a numpy array. Encoding is done by
    ``src/main/matlab/dataset/wu2/generate_audio_files.m``"""
    assert isinstance(b, bytes)

    tmp: int = int(len(b) / 4)
    return np.array(struct.unpack(_INDIANNESS + str(tmp) + "f", b))


def _read_bytes(io: IO, i: int, n: int) -> bytes:
    """
    Read a section of bytes.

    :param io: The IO stream to read bytes from
    :param i: The position of the first byte to read
    :param n: The number of bytes to read
    :return: The bytes
    """
    io.seek(i)
    return io.read(n)


def read_samples(io: IO, i: int, n: int) -> np.ndarray:
    """
    Read a section of samples.

    :param io: The IO stream to read the samples from
    :param i: The index of the first sample to read
    :param n: The number of samples to read
    :return: The samples
    """
    b: bytes = _read_bytes(io, i * _SAMPLE_BYTES, n * _SAMPLE_BYTES)

    return _decode_bytes(b)
