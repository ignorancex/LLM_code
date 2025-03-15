from pathlib import Path

from scipy.io import loadmat

import globalconfig as g_cfg
from dataset.commons import SampleType


def fs2str(fs_hz: float):
    """
    Convert a sampling frequency from float to string.

    :param fs_hz: the sampling frequency (in Hz).
    :return: the sampling frequency in a string (in kHz).
    """

    return str(int(fs_hz / 1000)) + 'kHz'


def get_database_csv_filename(sample_type: SampleType, fs_hz: int):
    """
    Returns the filename for a database csv file. Used for metadata loading.

    :param sample_type: Used to select between chews and bouts.
    :param fs_hz: The sampling frequency of audio to use.
    :return: The path of the csv file.
    """
    assert isinstance(sample_type, SampleType)
    assert isinstance(fs_hz, int)

    if sample_type is SampleType.CHEW:
        x = 'chews'
    elif sample_type is SampleType.BOUT:
        x = 'bouts'
    else:
        raise ValueError('Unsupported load_type: ' + str(sample_type))

    return g_cfg.get_generated_path() / Path(x + '/' + x + '_exported_' + fs2str(fs_hz) + '/' + x + '_database.csv')


def load_x_file(filename: Path):
    """
    Loads an audio signal (chew or bout) from a mat-file.

    :param filename: The mat-file name with full path.
    :return: The audio.
    """
    assert isinstance(filename, Path)
    if not filename.exists():
        raise RuntimeError("File not found: " + str(filename))

    x = loadmat(str(filename))['x']

    return x
