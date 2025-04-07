from typing import List, Dict, Iterable, Union
import os
from os import PathLike
import logging
from logging import Filter
import sys

__all__ = ["prepare_global_logging"]


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


FILE_FRIENDLY_LOGGING: bool = False


def prepare_global_logging(serialization_dir: Union[str, PathLike],
                           rank: int = 0,
                           world_size: int = 1,
                           log_name: str = "out") -> None:
    root_logger = logging.getLogger()

    normal_format = '[%(levelname)8s] %(message)s'
    error_format = '[%(asctime)s - %(levelname)8s - %(name)s] %(message)s'

    verbose_format = '[%(asctime)s - %(levelname)8s - %(name)s] %(message)s'

    # create handlers
    if world_size == 1:
        log_file = os.path.join(serialization_dir, f"{log_name}.log")
    else:
        log_file = os.path.join(serialization_dir,
                                f"{log_name}_worker{rank}.log")
        normal_format = f"{rank} | {normal_format}"
        verbose_format = f"{rank} | {verbose_format}"
        error_format = f"{rank} | {error_format}"

    normal_format = logging.Formatter(fmt=normal_format)
    error_format = logging.Formatter(fmt=error_format,
                                     datefmt='%Y-%m-%d %H:%M:%S')
    verbose_format = logging.Formatter(fmt=verbose_format,
                                       datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setFormatter(verbose_format)
    stderr_handler.setFormatter(error_format)

    stdout_handler = logging.StreamHandler(sys.stdout)

    stdout_handler.setFormatter(normal_format)
    level_name = "INFO"
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(LEVEL)
    stdout_handler.addFilter(
        ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(logging.DEBUG)

    # put all the handlers on the root logger
    root_logger.addHandler(file_handler)
    if rank == 0:
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

    # write uncaught exceptions to the logs
    def excepthook(exctype, value, traceback):
        # For a KeyboardInterrupt, call the original exception handler.
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception",
                             exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook
