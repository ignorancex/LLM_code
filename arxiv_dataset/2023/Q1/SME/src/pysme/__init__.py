# -*- coding: utf-8 -*-
__file_ending__ = ".sme"

# Load correct version string
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Add output to the console
import logging

import colorlog
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = TqdmLoggingHandler()
console.setLevel(logging.INFO)
console.setFormatter(
    colorlog.ColoredFormatter("%(log_color)s%(levelname)s - %(message)s")
)
logger.addHandler(console)

# Download library if it does not exist
import os.path
from ctypes import cdll

from .smelib import libtools

libfile = libtools.get_full_libfile()
if not os.path.exists(libfile):
    libtools.download_libsme()

try:
    cdll.LoadLibrary(libfile)
    from .smelib import _smelib
except:
    libtools.compile_interface()

# Provide submodules to the outside
__all__ = [
    "util",
    "abund",
    "atmosphere",
    "broadening",
    "continuum_and_radial_velocity",
    "cwrapper",
    "echelle",
    "iliffe_vector",
    "linelist",
    "nlte",
    "sme_synth",
    "sme",
    "solve",
    "uncertainties",
    "smelib",
]
