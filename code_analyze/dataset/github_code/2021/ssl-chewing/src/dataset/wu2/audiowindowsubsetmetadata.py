from pathlib import Path
from typing import List

from dataset.wu2.binaryaudioutils import get_file_length
from dataset.wu2.loaders import AudioWindowLoaderConf, get_binary_audio_file
from dataset.wu2.wu2metadata import WU2Metadata
from utilities.bufferutils import get_noof_windows
from utilities.matlabutils import find_unique, strcmp
from utilities.typingutils import is_typed_list


def get_noof_audio_windows_per_session(md: WU2Metadata, conf: AudioWindowLoaderConf) -> List[int]:
    assert isinstance(md, WU2Metadata)
    assert isinstance(conf, AudioWindowLoaderConf)

    noof_windows: List[int] = []
    for session_idx in md.session_idxs:
        file: Path = get_binary_audio_file(session_idx, conf.tag)
        n: int = get_noof_windows(get_file_length(file), conf.wsize, conf.wstep)
        noof_windows.append(n)

    return noof_windows


def get_noof_audio_windows_per_partition(md: WU2Metadata, noof_per_session: List[int]) -> List[int]:
    assert isinstance(md, WU2Metadata)
    assert is_typed_list(noof_per_session, int)
    assert len(noof_per_session) == len(md.session_md)

    noof_per_part: List[int] = [0] * len(md.partitions)

    for i, session_md in enumerate(md.session_md):
        idx: int = find_unique(strcmp(md.partitions, session_md.user_name))
        noof_per_part[idx] += noof_per_session[i]

    return noof_per_part


class AudioWindowSubsetMetadata:
    def __init__(self, md: WU2Metadata, conf: AudioWindowLoaderConf):
        assert isinstance(md, WU2Metadata)
        assert isinstance(conf, AudioWindowLoaderConf)

        self._md: WU2Metadata = md
        self._conf: AudioWindowLoaderConf = conf
        self._noof_per_session: List[int] = get_noof_audio_windows_per_session(md, conf)
        self._noof_per_partition: List[int] = get_noof_audio_windows_per_partition(md, self._noof_per_session)

    def get_noof_samples(self) -> int:
        return sum(self._noof_per_partition)

    def get_global_idxs(self, partition_idx: int) -> List[int]:
        assert isinstance(partition_idx, int)

        offset: int = sum(self._noof_per_partition[0:partition_idx])

        return [i for i in range(offset, offset + self._noof_per_partition[partition_idx])]
