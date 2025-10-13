"""
Central logging configuration for the da_gp project.

This module provides consistent logging setup across all CLI tools and scripts.
Use setup_logging() in CLI entry points and get_logger() in all modules.
"""

import logging
import sys


def setup_logging(level="WARNING", json=False):
    """Configure root logger for CLI applications.

    Args:
        level: Log level as string or int (default: "WARNING")
        json: If True, use JSON formatting for structured logs
    """
    root = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    # Set logging level
    root.setLevel(level.upper() if isinstance(level, str) else level)

    # Create handler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)

    # Configure formatter
    if json:
        fmt = '{"time":"%(asctime)s","lvl":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


def get_logger(name):
    """Get a logger instance for the given module name.

    Args:
        name: Usually __name__ from the calling module

    Returns:
        logging.Logger instance
    """
    return logging.getLogger(name)
