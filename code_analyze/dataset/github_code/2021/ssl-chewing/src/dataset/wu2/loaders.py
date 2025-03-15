from pathlib import Path
from typing import Optional

import numpy as np

import globalconfig as g_conf
from dataset.template.baseloader import BaseFileLoader
from dataset.wu2.binaryaudioutils import get_file_length, read_samples
from utilities.bufferutils import get_noof_windows
from utilities.commonutils import sec2samples
from utilities.numpyutils import ensure_numpy_2d_vector
from utilities.typingutils import is_int


def get_binary_audio_file(session_idx: int, tag: str) -> Path:
    """
    Convenience function that returns the file name of a binary audio file based on its index and tag.

    :param session_idx: The session index of the file. This is the global index, as assigned by the metadata constructor
    :param tag: Which version to load, e.g. "2khz_hpf20hz"
    """
    assert isinstance(session_idx, int)
    assert isinstance(tag, str)

    file: Path = g_conf.get_generated_path() / "wu2" / "audio_preprocessed" / tag / (str(session_idx) + ".bin")
    assert file.exists()

    return file


class AudioWindowLoaderConf:
    def __init__(self, tag: str, fs: float, wsize_sec: float, wstep_sec: float, max_noof_windows: int = None):
        assert isinstance(tag, str)
        assert isinstance(wsize_sec, float)
        assert isinstance(wstep_sec, float)
        assert isinstance(max_noof_windows, int) or max_noof_windows is None
        assert 0 < wsize_sec
        assert 0 < wstep_sec
        assert 0 < fs

        self.tag: str = tag
        self.wsize_sec: float = wsize_sec
        self.wstep_sec: float = wstep_sec
        self.fs: float = fs

        self.wsize: int = sec2samples(wsize_sec, fs)
        self.wstep: int = sec2samples(wstep_sec, fs)
        self.max_noof_windows: Optional[int] = max_noof_windows


class AudioWindowLoader(BaseFileLoader):
    def __init__(self, session_idx: int, conf: AudioWindowLoaderConf):
        """
        Create a loader for a single (binary) audio file.

        :param session_idx: The session index of the file. This is the global index, as assigned by the metadata
                            constructor
        :param conf: The loader configuration
        """
        assert isinstance(session_idx, int)
        assert isinstance(conf, AudioWindowLoaderConf)

        self.file: Path = get_binary_audio_file(session_idx, conf.tag)

        super().__init__(self.file)

        self.conf: AudioWindowLoaderConf = conf

        self.length: int = get_noof_windows(get_file_length(self.file), conf.wsize, conf.wstep)
        if conf.max_noof_windows is not None:
            self.length = min(self.length, conf.max_noof_windows)

    def load(self, i: int) -> np.ndarray:
        """Loads the ``i``-th window from the file."""
        assert is_int(i)
        assert 0 <= i < self.length

        if self._io is None:
            raise ValueError("IO is not opened")

        return ensure_numpy_2d_vector(read_samples(self._io, i * self.conf.wstep, self.conf.wsize))
