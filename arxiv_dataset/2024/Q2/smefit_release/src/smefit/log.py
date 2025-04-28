# -*- coding: utf-8 -*-
import inspect
import logging

import rich.align
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)


def setup_console(logfile):
    global console
    if logfile is not None:
        log_object = open(logfile, "w", encoding="utf-8")
    else:
        log_object = None
    console = Console(
        file=log_object,
    )


def print_banner():
    banner = rich.align.Align(
        rich.panel.Panel.fit(
            inspect.cleandoc(
                r"""
                     ____  __  __ _____ _____ _ _____
                    / ___||  \/  | ____|  ___(_)_   _|
                    \___ \| |\/| |  _| | |_  | | | |
                     ___) | |  | | |___|  _| | | | |
                    |____/|_|  |_|_____|_|   |_| |_|

                A Standard Model Effective Field Theory Fitter
                """
            ),
            rich.box.SQUARE,
            padding=1,
            style="magenta",
        ),
        "center",
    )
    console.print(banner)
