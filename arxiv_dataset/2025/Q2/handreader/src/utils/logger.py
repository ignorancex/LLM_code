import logging

import pyrootutils

ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with the given directory and log level.

    Args:
        log_dir (str): Directory where log files will be saved.
        log_level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    return logger


LOGGER = setup_logger()
