import os
import logging

_default_postprocessed_phase_filename_str = "phases_{}_fbest_{}_fhigh_{}_flow_{}.hdf5"

# Create a folder to store postprocessed phases
_default_postprocessed_phase_dir = os.path.expanduser("~/.phazap")
if not os.path.exists(_default_postprocessed_phase_dir):
    try:
        os.makedirs(_default_postprocessed_phase_dir)
    except:
        # Fail silently
        _default_postprocessed_phase_dir = os.getcwd()

# Create a logger for phazap
# The following code is modified from bilby_pipe.utils
def setup_logger(prog_name, log_level="INFO"):
    """Setup logging output: call at the start of the script to use

    Parameters
    ----------
    prog_name: str
        Name of the program
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError(f"log_level {log_level} not understood")
    else:
        level = int(log_level)

    logger = logging.getLogger(prog_name)
    logger.propagate = False
    logger.setLevel(level)

    streams = [isinstance(h, logging.StreamHandler) for h in logger.handlers]
    if len(streams) == 0 or not all(streams):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
            )
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

_prog_ = "phazap"
setup_logger(_prog_)
logger = logging.getLogger(_prog_)

from .postprocess_phase import postprocess_phase
from .phazap import phazap, phazap_summary