# File: markovianess/utils.py
"""
utils.py
--------
Common utilities for the markovianess package, including logging setup and
miscellaneous helper functions.
"""

import logging
import os
import sys


###############################################################################
# Logging Configuration
###############################################################################
def setup_logger(log_dir="logs", log_filename="app.log", level=logging.INFO):
    """
    Creates a dedicated logger named "markovianess" that writes to both
    console (stdout) and a file in `log_dir/log_filename`.
    """
    logger = logging.getLogger("markovianess")
    logger.setLevel(level)

    # Format for log messages
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - markovianess: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create a global logger instance (you can import and use it in other scripts)
logger = setup_logger()


###############################################################################
# Example utility: rolling-average smoothing (similar to ones used in plotting)
###############################################################################
def smooth_curve(data, window=10):
    """
    Smooths a 1D array 'data' by averaging over a rolling window of size 'window'.
    Returns a new 1D array of the same length (with edges partially smoothed).
    """
    import numpy as np
    if window <= 1:
        return data
    out = np.copy(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    for i in range(len(data)):
        start = max(0, i - window + 1)
        length = i - start + 1
        out[i] = (cumsum[i + 1] - cumsum[start]) / float(length)
    return out


###############################################################################
# more shared utilities below if desired.
###############################################################################
